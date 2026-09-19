import type { Redis } from 'ioredis';

export interface Turn {
    role: 'user' | 'assistant';
    content: string;
}

/** 每个会话保留的最大轮数（user+assistant 各算一条） */
const MAX_TURNS = 20;
/** 会话空闲过期时间（秒）。每次追加消息时滑动续期。 */
const SESSION_TTL_SECONDS = Number(process.env.SESSION_TTL_SECONDS ?? 1800);

export interface SessionStore {
    readonly backend: 'redis' | 'memory';
    getHistory(sessionId: string): Promise<Turn[]>;
    append(sessionId: string, turns: Turn[]): Promise<void>;
    close(): Promise<void>;
}

/**
 * 进程内兜底实现。
 *
 * 代价是显式接受的：重启丢会话、多实例之间不共享。
 * 但它保证 demo 路径**不需要任何外部服务**——这是本项目一贯的取舍
 * （同样的理由否决了需要常驻服务的 ChromaDB）。
 * 与原来的裸 Map 相比，这里补上了两件原实现缺的东西：条数上限 + 过期清理。
 */
class MemorySessionStore implements SessionStore {
    readonly backend = 'memory' as const;
    private sessions = new Map<string, { turns: Turn[]; expiresAt: number }>();

    private sweep() {
        const now = Date.now();
        for (const [id, s] of this.sessions) {
            if (s.expiresAt <= now) this.sessions.delete(id);
        }
    }

    async getHistory(sessionId: string): Promise<Turn[]> {
        this.sweep();
        return this.sessions.get(sessionId)?.turns ?? [];
    }

    async append(sessionId: string, turns: Turn[]): Promise<void> {
        this.sweep();
        const existing = this.sessions.get(sessionId)?.turns ?? [];
        const merged = [...existing, ...turns].slice(-MAX_TURNS);
        this.sessions.set(sessionId, {
            turns: merged,
            expiresAt: Date.now() + SESSION_TTL_SECONDS * 1000,
        });
    }

    async close(): Promise<void> {
        this.sessions.clear();
    }
}

/**
 * Redis 实现。
 *
 * 数据结构选 LIST 而不是 STRING/HASH：这个场景要的就是"追加 + 定长滑窗"，
 * RPUSH + LTRIM 正好是这个语义，不需要把整段历史读出来、截断、再写回去。
 * 每次写入后 EXPIRE 做滑动续期——TTL 顺带解决了原实现"Map 只增不减"的无界内存问题。
 */
class RedisSessionStore implements SessionStore {
    readonly backend = 'redis' as const;
    constructor(private redis: Redis) {}

    private key(sessionId: string) {
        return `session:${sessionId}`;
    }

    async getHistory(sessionId: string): Promise<Turn[]> {
        const raw = await this.redis.lrange(this.key(sessionId), 0, -1);
        const turns: Turn[] = [];
        for (const item of raw) {
            try {
                turns.push(JSON.parse(item));
            } catch {
                // 单条坏数据不该毁掉整个会话
            }
        }
        return turns;
    }

    async append(sessionId: string, turns: Turn[]): Promise<void> {
        if (turns.length === 0) return;
        const key = this.key(sessionId);
        // pipeline 让三条命令一次往返；RPUSH/LTRIM/EXPIRE 各自原子，
        // 这里不需要 Lua——顺序执行的中间态（超长一瞬）无害。
        await this.redis
            .multi()
            .rpush(key, ...turns.map(t => JSON.stringify(t)))
            .ltrim(key, -MAX_TURNS, -1)
            .expire(key, SESSION_TTL_SECONDS)
            .exec();
    }

    async close(): Promise<void> {
        await this.redis.quit();
    }
}

/**
 * 按 REDIS_URL 是否配置来决定后端。
 * 未配置 → 内存兜底，不报错、不阻塞启动（demo 默认路径）。
 * 配置了但连不上 → 同样降级到内存，并打印告警，不让服务起不来。
 */
export async function createSessionStore(): Promise<SessionStore> {
    const url = process.env.REDIS_URL;
    if (!url) {
        console.log('[session] REDIS_URL 未配置，使用进程内会话存储（重启丢会话、不支持多实例）');
        return new MemorySessionStore();
    }
    try {
        const { Redis: IORedis } = await import('ioredis');
        const redis = new IORedis(url, {
            maxRetriesPerRequest: 2,
            lazyConnect: true,
            connectTimeout: 3000,
        });
        await redis.connect();
        await redis.ping();
        console.log(`[session] 已连接 Redis（${url.replace(/:[^:@]*@/, ':***@')}），会话带 TTL ${SESSION_TTL_SECONDS}s`);
        return new RedisSessionStore(redis);
    } catch (e) {
        console.warn('[session] Redis 连接失败，降级为进程内存储:', (e as Error).message);
        return new MemorySessionStore();
    }
}
