import MCPClient from "./MCPClient.js";
import ChatOpenAI, { ToolCall } from "./ChatOpenAI.js";
import { Tool } from "@modelcontextprotocol/sdk/types.js";

/** 工具返回内容的截断上限。fetch 一个网页可能几万字，原样塞回上下文会直接爆 token。 */
const MAX_TOOL_RESULT_CHARS = 4000;
/** 工具循环的最大轮数。模型抖动时 while(true) 会无限烧 token。 */
const DEFAULT_MAX_ITERATIONS = 8;
/** 单个工具调用的超时。MCP 子进程卡死不能拖死整个请求。 */
const DEFAULT_TOOL_TIMEOUT_MS = 30_000;

export interface AgentEvent {
    type: 'tool_start' | 'tool_end' | 'tool_error' | 'iteration' | 'warning';
    message: string;
    detail?: unknown;
}

export interface AgentOptions {
    model: string;
    mcpClients: MCPClient[];
    systemPrompt?: string;
    context?: string;
    history?: { role: 'user' | 'assistant'; content: string }[];
    maxIterations?: number;
    toolTimeoutMs?: number;
    onToken?: (token: string) => void;
    onEvent?: (event: AgentEvent) => void;
    signal?: AbortSignal;
}

export default class Agent {
    private mcpClients: MCPClient[];
    private llm: ChatOpenAI | null = null;
    private opts: AgentOptions;
    /** 记录实际 init 成功的 client，close 时只关这些 */
    private initialized: MCPClient[] = [];

    constructor(opts: AgentOptions) {
        this.opts = opts;
        this.mcpClients = opts.mcpClients;
    }

    /**
     * 初始化。
     *
     * 降级策略：单个 MCP server 起不来（没装 uvx / npx 拉不到包 / 离线）时，
     * 记 warning 并**继续以无工具模式运行**，而不是让整个请求 500。
     * RAG 检索已经拿到法条了，没有工具照样能回答——降级链不能在最外层断掉。
     */
    async init() {
        const tools: Tool[] = [];
        for (const client of this.mcpClients) {
            try {
                await client.init();
                this.initialized.push(client);
                tools.push(...client.getTools());
            } catch (e) {
                this.opts.onEvent?.({
                    type: 'warning',
                    message: `MCP server "${client.getName()}" 启动失败，本次以无工具模式继续`,
                    detail: (e as Error).message,
                });
                console.warn(`[agent] MCP "${client.getName()}" init 失败，降级为无工具模式:`, (e as Error).message);
            }
        }
        if (tools.length === 0) {
            console.warn('[agent] 无可用工具，仅依赖本地知识库回答');
        }
        this.llm = new ChatOpenAI({
            model: this.opts.model,
            systemPrompt: this.opts.systemPrompt,
            tools,
            context: this.opts.context,
            history: this.opts.history,
            onToken: this.opts.onToken,
        });
    }

    /** 只关闭真正 init 成功的 client，避免对未连接的 client 调 close */
    async close() {
        const targets = this.initialized;
        this.initialized = [];
        for (const client of targets) {
            try {
                await client.close();
            } catch (e) {
                console.warn(`[agent] 关闭 MCP "${client.getName()}" 失败:`, (e as Error).message);
            }
        }
    }

    getUsage() {
        return this.llm?.getUsage() ?? { promptTokens: 0, completionTokens: 0 };
    }

    /**
     * 跑工具循环并返回最终回答。
     *
     * 注意：这里**不再** close()。生命周期由调用方掌握——
     * 原实现在 invoke() 结尾 close()，而调用方又 close() 了一次，属于重复关闭。
     */
    async invoke(prompt: string): Promise<string> {
        if (!this.llm) throw new Error('Agent 未初始化，请先调用 init()');

        const maxIterations = this.opts.maxIterations ?? DEFAULT_MAX_ITERATIONS;
        let response = await this.llm.chat(prompt, this.opts.signal);

        for (let iteration = 1; iteration <= maxIterations; iteration++) {
            if (response.toolCalls.length === 0) {
                return response.content;
            }
            this.opts.onEvent?.({ type: 'iteration', message: `第 ${iteration}/${maxIterations} 轮工具调用` });

            for (const toolCall of response.toolCalls) {
                await this.runOneTool(toolCall);
            }
            response = await this.llm.chat(undefined, this.opts.signal);
        }

        // 到达上限仍在要求调用工具：不再继续，返回已有内容并说明原因
        this.opts.onEvent?.({
            type: 'warning',
            message: `已达工具调用上限（${maxIterations} 轮），停止循环`,
        });
        return response.content || `（已达工具调用上限 ${maxIterations} 轮，未能得到最终回答。）`;
    }

    private async runOneTool(toolCall: ToolCall) {
        const llm = this.llm!;
        const name = toolCall.function.name;

        const mcp = this.initialized.find(client =>
            client.getTools().some((t) => t.name === name)
        );
        if (!mcp) {
            llm.appendToolResult(toolCall.id, `Tool "${name}" not found or its MCP server is unavailable.`);
            return;
        }

        // 流式拼出来的 arguments 可能是不完整 JSON，裸 JSON.parse 会让整个请求 500
        let args: Record<string, unknown>;
        try {
            args = JSON.parse(toolCall.function.arguments || '{}');
        } catch (e) {
            const msg = `Invalid tool arguments (not valid JSON): ${toolCall.function.arguments?.slice(0, 200)}`;
            this.opts.onEvent?.({ type: 'tool_error', message: `工具 ${name} 参数解析失败`, detail: msg });
            llm.appendToolResult(toolCall.id, msg);
            return;
        }

        this.opts.onEvent?.({ type: 'tool_start', message: `调用工具 ${name}`, detail: args });

        try {
            const result = await withTimeout(
                mcp.callTool(name, args),
                this.opts.toolTimeoutMs ?? DEFAULT_TOOL_TIMEOUT_MS,
                `工具 ${name} 超时`
            );
            const serialized = truncate(JSON.stringify(result), MAX_TOOL_RESULT_CHARS);
            this.opts.onEvent?.({ type: 'tool_end', message: `工具 ${name} 返回` });
            llm.appendToolResult(toolCall.id, serialized);
        } catch (e) {
            // 工具失败要作为 tool 消息回灌给模型（让它换个思路），而不是抛出中断整个请求
            const msg = `Tool "${name}" failed: ${(e as Error).message}`;
            this.opts.onEvent?.({ type: 'tool_error', message: msg });
            llm.appendToolResult(toolCall.id, msg);
        }
    }
}

function truncate(text: string, max: number): string {
    if (text.length <= max) return text;
    return text.slice(0, max) + `\n...[已截断，原长度 ${text.length} 字符]`;
}

function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(message)), ms);
        promise.then(
            (v) => { clearTimeout(timer); resolve(v); },
            (e) => { clearTimeout(timer); reject(e); }
        );
    });
}
