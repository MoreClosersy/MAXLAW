import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { Tool } from "@modelcontextprotocol/sdk/types.js";

export default class MCPClient {
    private mcp: Client;
    private name: string;
    private command: string;
    private args: string[];
    private transport: StdioClientTransport | null = null;
    private tools: Tool[] = [];
    private connected = false;

    constructor(name: string, command: string, args: string[], version?: string) {
        this.name = name;
        this.mcp = new Client({ name, version: version || "0.0.1" });
        this.command = command;
        this.args = args;
    }

    getName() {
        return this.name;
    }

    isConnected() {
        return this.connected;
    }

    async init() {
        if (this.connected) return;
        await this.connectToServer();
    }

    /** 幂等关闭：未连接时直接返回；同时显式关掉 transport（子进程），不依赖 SDK 内部实现 */
    async close() {
        if (!this.connected) return;
        this.connected = false;
        this.tools = [];
        try {
            await this.mcp.close();
        } finally {
            try {
                await this.transport?.close();
            } catch {
                // transport 可能已随 mcp.close() 一起关闭，忽略
            }
            this.transport = null;
        }
    }

    getTools(): Tool[] {
        return this.tools;
    }

    callTool(name: string, params: Record<string, unknown>) {
        if (!this.connected) {
            return Promise.reject(new Error(`MCP "${this.name}" 未连接`));
        }
        return this.mcp.callTool({ name, arguments: params });
    }

    private async connectToServer() {
        try {
            this.transport = new StdioClientTransport({
                command: this.command,
                args: this.args,
            });
            await this.mcp.connect(this.transport);
            const toolsResult = await this.mcp.listTools();
            this.tools = toolsResult.tools.map((tool) => ({
                name: tool.name,
                description: tool.description,
                inputSchema: tool.inputSchema,
            }));
            this.connected = true;
            console.log(`[mcp] "${this.name}" 已连接，工具:`, this.tools.map(({ name }) => name).join(', '));
        } catch (e) {
            // init 失败要清理已创建的 transport，否则子进程泄漏
            try { await this.transport?.close(); } catch { /* ignore */ }
            this.transport = null;
            this.connected = false;
            throw e;
        }
    }
}
