# 08 — 微分、偏微分、勾配、連鎖律

## この章で作るもの

一変数関数の数値微分と、多変数関数の gradient を central finite difference で実装します。
対象コードは `src/main/scala/learnai/math/Calculus.scala` です。

この実装は大規模な学習には遅過ぎますが、次章以降で作る自動微分が正しいかを独立した方法で
検査する基準になります。

## 関数は入力を出力へ対応させる

関数 \(f\) が実数 \(x\) を受け取り実数 \(y\) を返すことを次のように書きます。

\[
f:\mathbb{R}\to\mathbb{R},\qquad y=f(x)
\]

例 \(f(x)=x^2\) なら、`x = 3` のとき `y = 9` です。グラフの一点だけでなく、「入力を少し
動かしたとき出力がどれだけ変わるか」が最適化に必要です。

## 平均変化率から瞬間の傾きへ

\(x\) から \(x+h\) までの平均変化率は次です。

\[
\frac{f(x+h)-f(x)}{h}
\]

\(h\) を 0 に近づけた極限が derivative（導関数）です。

\[
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
\]

これは「入力を非常に少し増やしたとき、出力が入力の何倍くらい変わるか」を表します。
単位も重要です。loss を weight で微分した値の単位は `loss / weight` です。

## 基本的な微分規則

| 関数 | 導関数 |
| --- | --- |
| \(c\) | \(0\) |
| \(x\) | \(1\) |
| \(x^n\) | \(nx^{n-1}\) |
| \(f+g\) | \(f'+g'\) |
| \(fg\) | \(f'g+fg'\) |
| \(e^x\) | \(e^x\) |
| \(\log x\) | \(1/x\) |
| \(\tanh x\) | \(1-\tanh^2x\) |

たとえば \(f(x)=x^2\) なら \(f'(x)=2x\)、`x = 3` で傾きは `6` です。

## 数値微分

極限を computer で直接計算できないため、小さい有限値 \(h\) を使います。片側だけを使うより誤差を
減らす central difference は次です。

\[
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}
\]

```scala
val above = function(at + step)
val below = function(at - step)
(above - below) / (2.0 * step)
```

`h` が大き過ぎると曲線の離れた二点の傾きになり、小さ過ぎると浮動小数点の桁落ちが増えます。
既定値 `1e-5` は万能な正解ではありません。gradient check では複数の step も試します。

## 多変数関数と偏微分

neural network の loss は、全 parameter を要素に持つ vector から一つの scalar を返す関数と
見なせます。

\[
L:\mathbb{R}^n\to\mathbb{R}
\]

一つの変数 \(x_i\) だけを動かし、他を固定して測る傾きが partial derivative（偏微分）です。

\[
\frac{\partial L}{\partial x_i}
\]

全偏微分を並べた vector が gradient です。

\[
\nabla L(\boldsymbol{x})=
\begin{bmatrix}
\frac{\partial L}{\partial x_1} \\
\vdots \\
\frac{\partial L}{\partial x_n}
\end{bmatrix}
\]

shape は loss が scalar でも gradient は parameter と同じ `[n]` です。

## gradient は最も増加する方向

unit vector \(\boldsymbol{u}\) 方向の変化率は directional derivative です。

\[
D_{\boldsymbol{u}}L=\nabla L\cdot\boldsymbol{u}
\]

gradient と同じ方向で増加率が最大、反対の \(-\nabla L\) 方向で最も減少します。これが gradient
descent で parameter から gradient を引く理由です。

## 連鎖律

関数を合成した \(y=f(g(x))\) の微分は、経路上の局所的な微分を掛けます。

\[
\frac{dy}{dx}=\frac{dy}{dg}\frac{dg}{dx}
\]

例として \(g(x)=x^2\)、\(f(g)=3g+1\) なら、

\[
\frac{dy}{dx}=3\times 2x=6x
\]

neural network は層を何度も合成した関数です。backpropagation は出力から入力へ連鎖律を効率よく
適用し、共有された中間計算の微分を再利用します。

## 数値微分の計算量

parameter が \(n\) 個なら、central difference は loss の forward 計算を `2n` 回行います。
10 億 parameter なら現実的ではありません。reverse-mode automatic differentiation は、概ね
数回の forward 相当の計算で全 gradient を求めます。

数値微分を捨てるのではなく、小さい入力で自動微分と比較する **gradient check** に使います。
同じ bug を共有しない別方式が reference になります。

## 実行と確認

```console
$ nix develop -c sbt 'runMain learnai.math.runGradientLab'
point:    VectorD(2.0, -1.0)
gradient: VectorD(1.0..., 4.0...)
expected: VectorD(1.0, 4.0)

$ nix develop -c sbt test
```

## 演習

1. \(f(x)=3x^2+2x+1\) の導関数を手で求め、`x = 2` で数値微分と比べてください。
2. step を `1e-1` から `1e-12` まで変え、誤差を表にしてください。
3. \(f(x,y)=x^2+y^2\) の gradient を手で求めてください。
4. gradient の反対方向へ小さく動くと関数値が減ることを実験してください。
5. `Calculus.gradient` が `2n` 回 function を呼ぶことを counter で確認してください。

## 完了条件

- derivative を「入力を少し変えたときの出力の変化率」と説明できる
- partial derivative と gradient の shape を説明できる
- gradient の反対方向で loss が減る理由を説明できる
- 連鎖律を二つの合成関数へ適用できる
- 数値微分を学習本体でなく gradient check に使う理由を説明できる
- `CalculusSuite` が成功する
