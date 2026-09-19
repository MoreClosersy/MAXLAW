/**
 * `eval/abstentionCore.ts` 的测试。
 *
 * 这个文件守两件事：
 *
 * 1. **抽取没有改变行为**。`bigrams`/`coverage`/`auc` 是从 `eval/abstentionSignals.ts`
 *    逐字搬过来的，搬完 `pnpm eval:signals` 的输出逐字节一致——但那是一次性的验证，
 *    下次有人动这几个函数时没有东西拦着。这里把它们的行为钉住。
 *
 * 2. **二元信号的 AUC 不许被当成连续信号读**。判官是二元的，它的 AUC 完全由两个错误率决定，
 *    把它和 cross-encoder 的 0.938 并排比会读出一个不存在的结论。下面那条代数断言
 *    就是把这个警告变成可执行的东西——见 `auc` 的注释。
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bigrams, coverage, auc, scanThresholds } from '../../eval/abstentionCore.js';

// ---------------------------------------------------------------------------
// bigrams / coverage
// ---------------------------------------------------------------------------
test('bigrams: 剥掉标点和「第/条」', () => {
    // 中文没有词边界，bigram 是最稳的近似；「第…条」是引用格式，不是语义内容
    assert.deepEqual([...bigrams('第188条')].sort(), ['18', '88']);
    assert.deepEqual([...bigrams('时效')].sort(), ['时效']);
    // 标点被剥掉后剩下的两字仍然成 bigram
    assert.deepEqual([...bigrams('时效，期间')].sort(), ['效期', '时效', '期间']);
});

test('bigrams: 单字串没有 bigram', () => {
    assert.equal(bigrams('法').size, 0);
    assert.equal(bigrams('').size, 0);
});

test('coverage: 查询的 bigram 全在正文里 → 1', () => {
    assert.equal(coverage('诉讼时效', ['向人民法院请求保护民事权利的诉讼时效期间为三年。']), 1);
});

test('coverage: 毫不相干 → 0', () => {
    assert.equal(coverage('番茄炒蛋怎么做', ['诉讼时效期间为三年。']), 0);
});

test('coverage: 空查询 → 0 而不是 NaN', () => {
    // 分母是 qb.size。若忘了那个 size===0 的守卫，这里会返回 NaN，
    // 而 NaN 在后面的 filter(x => x < t) 里恒为 false——它会静默地"永不被拒答"。
    assert.equal(coverage('', ['诉讼时效']), 0);
    assert.equal(coverage('，。', ['诉讼时效']), 0);
});

// ---------------------------------------------------------------------------
// auc
// ---------------------------------------------------------------------------
test('auc: 完美分离 → 1，完全反转 → 0', () => {
    assert.equal(auc([0.9, 0.8], [0.2, 0.1]), 1, '域内全高于域外，信号完全可分');
    assert.equal(auc([0.1, 0.2], [0.8, 0.9]), 0, '域内全低于域外，信号方向反了');
});

test('auc: 同分布 → 0.5', () => {
    assert.equal(auc([0.5], [0.5]), 0.5, '全平局');
    assert.equal(auc([0.1, 0.9], [0.1, 0.9]), 0.5);
});

test('auc: 平局算 0.5 而不是 0', () => {
    // 判成 0 会让"所有分数都一样"这种最没用的信号看起来比抛硬币还差，
    // 从而被误读成"反相关"。0.5 才是"没有区分力"。
    assert.equal(auc([0.5, 0.5], [0.5, 0.5]), 0.5);
    assert.equal(auc([0.5], [0.5, 0.1]), 0.75, '一半平局一半赢');
});

test('auc: 空输入 → NaN，不是 0', () => {
    // 0 会被读成"完全反转"（一个很强的结论），NaN 才是"算不出来"。
    assert.ok(Number.isNaN(auc([], [0.5])), '没有正类时算不出 AUC');
    assert.ok(Number.isNaN(auc([0.5], [])), '没有负类时算不出 AUC');
    assert.ok(Number.isNaN(auc([], [])));
});

test('auc: 二元信号上恒等于 0.5 + (TPR − FPR)/2 —— 所以二元 AUC 不携带额外信息', () => {
    // 这条是**可执行的警告**，不是数学练习。
    // 判官只有 {可答, 不可答} 两个取值，它的 AUC 完全由两个错误率决定；
    // 把它和 cross-encoder 的 0.938 并排放在一张表里，
    // 会让人以为"判官 0.9xx vs 重排 0.938"是一次有意义的比较，其实不是：
    // 连续信号的 AUC 描述整条 ROC，二元信号只有**一个**操作点。
    const pos = [1, 1, 1, 0, 0];   // 域内：3 条"可答"、2 条"拒答"
    const neg = [1, 0, 0, 0, 0];   // 域外：1 条"可答"（漏拒）、4 条"拒答"

    const tpr = pos.filter(x => x === 1).length / pos.length;   // 0.6
    const fpr = neg.filter(x => x === 1).length / neg.length;   // 0.2
    assert.equal(tpr, 0.6);
    assert.equal(fpr, 0.2);
    assert.equal(auc(pos, neg), 0.5 + (tpr - fpr) / 2);
    assert.equal(auc(pos, neg), 0.7);
});

// ---------------------------------------------------------------------------
// scanThresholds
// ---------------------------------------------------------------------------
test('scanThresholds: 域外严格低于域内 → 域外全拒且零误拒', () => {
    const s = scanThresholds([0.1, 0.2], [0.5, 0.6]);
    assert.ok(s.allOod, '应当找到"域外全拒"的阈值');
    assert.equal(s.allOod!.falseRefusals, 0, '全部域外都低于全部域内时，全拒不该误伤域内');
    assert.ok(s.zeroFalseRefusal, '应当找到"零误拒"的阈值');
    assert.equal(s.zeroFalseRefusal!.caught, 2, '零误拒下能把 2 条域外全拦下');
});

test('scanThresholds: 上哨兵是承重的 —— 最大取值属于域外时必须还能全拒', () => {
    // 候选阈值取的是"相邻取值的中点"。若只在相邻中点里扫，当下确界是最大的那个取值
    // （这里是域外的 0.9）时，候选里最大的中点是 0.7，`x < 0.7` 永远够不到 0.9 那条，
    // 于是"域外全拒"这一行**永远算不出来**（或者报出一个假的更差结果）。
    // 补的那个 vals[last] + 1e-6 就是为它存在的。
    const s = scanThresholds([0.1, 0.9], [0.5]);
    assert.ok(s.allOod, '最大取值属于域外时，"域外全拒"仍然必须能算出来');
    // 阈值必须**超过**那个最大的取值（0.9）。这恰好就是上哨兵的值，
    // 所以这条断言同时在说："全拒这一行确实是靠哨兵才够到的"。
    // 去掉哨兵，最大候选是 0.7，`x < 0.7` 拦不下 0.9，allOod 直接是 null。
    assert.ok(s.allOod!.threshold > 0.9, `全拒阈值应当超过最大取值，实际 ${s.allOod!.threshold}`);
    // 代价如实记下来：要拦住 0.9 这条域外，0.5 那条域内必然被误伤
    assert.equal(s.allOod!.falseRefusals, 1, '阈值高过 0.5，域内的 0.5 会被一起拒掉');
});

test('scanThresholds: 混合分布下两个端点都是非平凡的', () => {
    // 回归测试：第一版把**已经 map 成数值**的数组又套了一次取值函数，
    // 取值函数作用在数字上返回 undefined，于是 caught/bad 恒为 0，
    // 整个扫描静默地"什么都没发现"，却照样打印出一张格式完好的表。
    // 这个构造里两个端点都必须严格落在中间，恒 0 的实现会立刻露馅。
    const ood = [0.05, 0.15, 0.25, 0.70];
    const dom = [0.20, 0.40, 0.60, 0.80];
    const s = scanThresholds(ood, dom);

    // 全拒要拦住 0.70，阈值就得高过 0.70，于是域内的 0.20/0.40/0.60 三条全被误伤。
    // 这就是"两端都很差"的具体样子：另一端（零误拒）只能拦下 2/4。
    assert.ok(s.allOod, '应当能全拒');
    assert.equal(s.allOod!.falseRefusals, 3, '全拒 0.70 的代价是误伤三条域内');

    assert.ok(s.zeroFalseRefusal, '应当有零误拒的操作点');
    // 零误拒要求阈值 ≤ 0.20，此时只有 0.05 和 0.15 被拦下（阈值取 0.175）
    assert.equal(s.zeroFalseRefusal!.caught, 2, '零误拒下只能拦下 0.05/0.15 两条');
    assert.ok(s.zeroFalseRefusal!.caught < ood.length, '零误拒不可能拦下全部——否则这个构造没意义');
    // 两端都非平凡：拦得多的那端误伤 3 条，不误伤的那端只拦住一半
    assert.ok(s.allOod!.falseRefusals > 0 && s.zeroFalseRefusal!.caught < ood.length,
        '恒 0 的实现会同时给出 falseRefusals=0 和 caught=0，这条把它们一起挡住');
});

test('scanThresholds: 空输入 → 两个端点都是 null，不抛异常', () => {
    // abstentionSignals.ts 里 vals[0] 在空数组上是 undefined，
    // 用它算出的是 NaN，后面整条扫描会在一堆 NaN 上跑完并打印出看似正常的结果。
    const s = scanThresholds([], []);
    assert.equal(s.allOod, null);
    assert.equal(s.zeroFalseRefusal, null);
});
