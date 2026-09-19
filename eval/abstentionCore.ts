/**
 * 拒答信号的纯逻辑：bigram 覆盖率、AUC、阈值端点扫描。
 *
 * ## 为什么单独一个文件
 *
 * `eval/abstentionSignals.ts` 顶层有 `await loadKnowledge()` 和一个跑遍全数据集的 for 循环，
 * 所以它**导不出任何东西**——import 它就会顺便加载整个语料、跑完整个数据集。
 * 新的 `eval/abstention.ts` 需要同一套 AUC/扫描实现，但不能继承那个副作用。
 *
 * 这与 `eval/judgeCore.ts` 的切分是同一条纪律，理由也一样：
 * **需要被测试的函数，不能住在有副作用模块的顶层。**
 *
 * ## 为什么不做成"合并两个 similarity"
 *
 * `judgeCore.ts` 里也有一个 `similarity`，那是 bigram Jaccard；这里的 `coverage` 是
 * "查询的 bigram 有多大比例出现在正文里"，而且 `bigrams` 会先剥掉标点和「第/条」。
 * 两者看着像，**换成同一个函数会静默改掉 §10.8 已经发布的数字**。
 * 宁可让两个名字长得像，也不要让一张已发布的表在没人注意的时候变了。
 */

/** 查询串的字符 bigram 集合（中文没有词边界，bigram 是最稳的近似） */
export function bigrams(s: string): Set<string> {
    const clean = s.replace(/[\s，。？！、；：""''（）《》第条]/g, '');
    const out = new Set<string>();
    for (let i = 0; i < clean.length - 1; i++) out.add(clean.slice(i, i + 2));
    return out;
}

/** 查询 bigram 在检索到的正文里出现的比例 */
export function coverage(q: string, docs: string[]): number {
    const qb = bigrams(q);
    if (qb.size === 0) return 0;
    const hay = new Set(bigrams(docs.join('')));
    let hit = 0;
    for (const g of qb) if (hay.has(g)) hit++;
    return hit / qb.size;
}

/**
 * 域内=正类。0.5 表示这个信号完全没有区分力。
 *
 * ⚠️ **二元信号上的 AUC 不携带额外信息**：对只有 {拒答, 不拒答} 两个取值的信号，
 * AUC ≡ 0.5 + (TPR − FPR)/2（Youden's J 的重标度）。也就是说它完全由那两个错误率决定，
 * 把它和连续信号的 AUC 并排比会读出一个不存在的结论——连续信号的 AUC 描述整条 ROC，
 * 而二元信号只有**一个**操作点。`src/tests/abstentionCore.test.ts` 把这条代数钉成了测试。
 */
export function auc(pos: number[], neg: number[]): number {
    if (pos.length === 0 || neg.length === 0) return NaN;
    let c = 0;
    for (const p of pos) for (const n of neg) c += p > n ? 1 : p === n ? 0.5 : 0;
    return c / (pos.length * neg.length);
}

export interface ThresholdScan {
    /** 域外全拒：所需的最小误拒数，以及取到的阈值 */
    allOod: { threshold: number; falseRefusals: number } | null;
    /** 零误拒：最多能拦下几条域外，以及取到的阈值 */
    zeroFalseRefusal: { threshold: number; caught: number } | null;
}

/**
 * 全域扫描阈值，找两个端点：
 * "域外全部拦下所需的最小误拒" 与 "零误拒能拦下几条域外"。
 *
 * 越小越该拒答，所以判据是 `x < t`。
 *
 * 候选阈值取相邻取值的中点，并**补一个比最大值更大的值**——直接用实际取值当阈值时，
 * `x < t` 会把该值自身排除在外，永远够不到"域外全拒"。这两个哨兵是承重的。
 *
 * 两个端点都是有意义的：它们分别是"宁可错杀"和"宁可放过"两种极端策略下的最好结果，
 * 而真正能上线的操作点在这两端之间——这也正是为什么光看这两端不够、需要 AUC。
 */
export function scanThresholds(ood: number[], dom: number[]): ThresholdScan {
    const vals = [...ood, ...dom].sort((a, b) => a - b);
    if (vals.length === 0) return { allOod: null, zeroFalseRefusal: null };

    const cands: number[] = [vals[0] - 1e-6];
    for (let i = 1; i < vals.length; i++) cands.push((vals[i - 1] + vals[i]) / 2);
    cands.push(vals[vals.length - 1] + 1e-6);

    let allOod: ThresholdScan['allOod'] = null;
    let zeroFalseRefusal: ThresholdScan['zeroFalseRefusal'] = null;
    for (const t of cands) {
        // 注意：ood/dom 里装的**已经是数值**，不能再套一次取值函数——
        // 那个函数作用在数字上会返回 undefined，于是 caught/bad 恒为 0，
        // 整个扫描静默地"什么都没发现"，输出的却是一张格式完好的表。第一版就是这么错的。
        const caught = ood.filter(x => x < t).length;
        const bad = dom.filter(x => x < t).length;
        if (caught === ood.length && (allOod === null || bad < allOod.falseRefusals)) {
            allOod = { threshold: t, falseRefusals: bad };
        }
        if (bad === 0 && (zeroFalseRefusal === null || caught > zeroFalseRefusal.caught)) {
            zeroFalseRefusal = { threshold: t, caught };
        }
    }
    return { allOod, zeroFalseRefusal };
}
