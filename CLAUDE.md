# MAP - Charting Student Math Misunderstandings

## コンペティション概要

### 目的
このコンペティションでは、学生の自由記述式回答から数学的な誤解（misconceptions）を正確に予測するNLPモデルの開発を目指します。このソリューションは、学生の説明に対して候補となる誤解を提案し、教師が学生の誤った思考を特定・対処しやすくすることで、数学学習の改善に貢献します。

### 背景
学生はしばしば数学的推論を説明するよう求められます。これらの説明は学生の思考に関する豊かな洞察を提供し、根本的な誤解（systematic incorrect ways of thinking）を明らかにすることがよくあります。

**誤解の例**:
- 学生は0.355が0.8より大きいと考えることがよくあります。これは、整数の知識を小数に誤って適用し、355が8より大きいと推論するためです。
- 学生は数学でさまざまな誤解を発展させます。時には以前の知識を新しい内容に誤って適用したり、新しい情報を理解しようとして誤解したりすることがあります。

### 課題
- 学生の説明に潜在的な誤解が含まれているかをタグ付けすることは診断的フィードバックに価値がありますが、時間がかかり、スケールすることが困難です
- 誤解は微妙で、特異性が異なり、学生の推論に新しいパターンが現れるにつれて進化します
- 事前学習済み言語モデルを使用する初期の試みは、問題の数学的内容の複雑さのため成功していません

### 主催者
- Vanderbilt University
- The Learning Agency
- データ提供: Eedi（9-16歳の学生向け教育テクノロジープラットフォーム）
- サポート: Gates Foundation, Walton Family Foundation

## 評価指標

### Mean Average Precision @ 3 (MAP@3)

$$\text{MAP}@3 = \frac{1}{U} \sum_{u=1}^{U} \sum_{k=1}^{\min(n,3)} P(k) \times rel(k)$$

ここで:
- $U$ は観測数
- $P(k)$ はカットオフ $k$ での精度
- $n$ は観測ごとに提出された予測数
- $rel(k)$ はランク $k$ のアイテムが関連する（正しい）ラベルの場合は1、そうでない場合は0を返す指標関数

**重要な点**:
- 各観測につき正しいラベルは1つのみ
- 一度正しいラベルがスコア付けされると、そのラベルはその観測に対して関連性がなくなり、追加の予測はスキップされます
- 例: 正しいラベルがAの場合、以下の予測はすべて平均精度1.0となります:
  - [A, B, C, D, E]
  - [A, A, A, A, A]
  - [A, B, A, C, A]

## データセット

### データの背景
Eediプラットフォームでは、学生は診断問題（Diagnostic Questions: DQs）に回答します：
- **形式**: 1つの正解と3つの誤答（distractors）を含む多肢選択問題
- **プロセス**: 
  1. 学生が多肢選択で回答を選択
  2. 時々、選択した回答の理由を説明する記述式回答を求められる
- **注意点**: 元の問題は画像形式で提示されていたが、数学的表現を含むすべての内容は人間参加型OCRプロセスで正確にテキスト化されている

### モデルが実行すべき3つのステップ
1. **選択された回答が正しいかを判定** (CategoryのTrue/False部分)
2. **説明に誤解が含まれているかを評価** (Correct/Misconception/Neither)
3. **特定の誤解を識別** (該当する場合のみ)

### データファイルの構造

#### train.csv / test.csv
| カラム名 | 説明 | データ型 | 備考 |
|---------|------|---------|------|
| `row_id` | 行識別子 | Integer | - |
| `QuestionId` | 一意の問題識別子 | Integer | - |
| `QuestionText` | 問題文のテキスト | String | OCRで抽出 |
| `MC_Answer` | 学生が選択した多肢選択の回答 | String | - |
| `StudentExplanation` | 学生の回答選択理由の説明 | String | 主要な分析対象 |
| `Category` | 回答と説明の関係の分類 | String | trainのみ |
| `Misconception` | 学生の説明で特定された数学的誤解 | String | trainのみ、該当なしは'NA' |

#### sample_submission.csv
| カラム名 | 説明 | データ型 | 備考 |
|---------|------|---------|------|
| `row_id` | 行識別子 | Integer | - |
| `Category:Misconception` | 予測値 | String | 最大3つまで、スペース区切り |

### データサイズ
- **test.csv**: 約3行（サンプル）
- **再実行テストデータ**: 約40,000行

### カテゴリの詳細
カテゴリは以下の組み合わせで構成されます：

**第1要素（正誤）**:
- `True_`: 多肢選択の回答が正しい
- `False_`: 多肢選択の回答が誤り

**第2要素（説明の評価）**:
- `_Correct`: 説明が正しい理解を示している
- `_Misconception`: 説明に誤解が含まれている
- `_Neither`: 説明が誤りだが特定の誤解パターンではない

**カテゴリの例**:
- `True_Misconception`: 正しい回答を選択したが、説明に誤解が含まれている
- `False_Correct`: 誤った回答を選択したが、説明は正しい理解を示している（ケアレスミスなど）
- `True_Correct`: 正しい回答を選択し、説明も正しい（理想的な状態）

### 提出フォーマットの詳細
```csv
row_id,Category:Misconception
36696,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
36697,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
36698,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
```

**注意事項**:
- カテゴリとMisconceptionはコロン（:）で連結
- Misconceptionが該当しない場合は'NA'を使用
- 予測は信頼度の高い順に最大3つまで
- 4つ目以降の予測は無視される

## タイムライン

- **2025年7月10日**: コンペティション開始
- **2025年10月8日**: 
  - エントリー締切（ルール承認期限）
  - チームマージ締切
- **2025年10月15日**: 最終提出締切

*すべての締切は対応する日のUTC 23:59*

## 賞金

### リーダーボード賞金
- 1位: $20,000
- 2位: $12,000
- 3位: $8,000
- 4位: $5,000
- 5位: $5,000
- 6位: $5,000

**総額: $55,000**

## コンペティション要件

### Code Competition
- **提出方法**: Notebookを通じて提出
- **実行時間制限**: 
  - CPU Notebook ≤ 9時間
  - GPU Notebook ≤ 9時間
- **インターネットアクセス**: 無効
- **外部データ**: 自由に公開されているデータ（事前学習済みモデルを含む）は使用可能
- **提出ファイル名**: `submission.csv`

## プロジェクト構成案

```
map-competition/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_submission.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── configs/
├── logs/
└── submissions/
```

## アプローチの検討事項

### 技術的課題
1. **数学的内容の複雑さ**: 従来の事前学習済み言語モデルでは対応困難
2. **誤解の微妙さ**: 誤解パターンは微妙で多様
3. **カテゴリ分類**: 3つのカテゴリ（正解、誤解、その他の誤り）への分類
4. **複数ラベル予測**: 各サンプルに対して最大3つの予測

### 推奨アプローチ
- **多段階分類アプローチ**: 
  1. まず正誤判定（True/False）
  2. 次に説明の評価（Correct/Misconception/Neither）
  3. 最後に具体的な誤解の特定
- **数学特化型の言語モデル**: MathBERT、数式認識に強いモデルの活用
- **特徴量エンジニアリング**:
  - 数学的表現の抽出
  - 問題文と学生説明の関連性スコア
  - 誤解パターンの辞書ベース特徴
- **アンサンブル手法**: 複数モデルの予測を組み合わせてMAP@3を最適化
- **データ拡張**: 数学的な同義表現の生成

## 次のステップ

1. **データダウンロードと初期確認**:
   - Kaggleからデータセットをダウンロード
   - train.csvのサイズとカラムの確認
   - カテゴリとMisconceptionの種類をリストアップ

2. **探索的データ分析（EDA）**: 
   - カテゴリの分布（True/False × Correct/Misconception/Neither）
   - Misconceptionの頻度分析（上位の誤解パターン）
   - 問題文と学生説明の長さの分布
   - 問題の難易度による誤解パターンの違い

3. **データ前処理**:
   - 数式や数学記号の正規化
   - テキストのクリーニング（OCRエラーの可能性を考慮）
   - カテゴリとMisconceptionの分離

4. **ベースライン構築**: 
   - ルールベースの簡単な分類器
   - TF-IDFベースの分類モデル
   - 事前学習済みBERTモデルの初期実装
   - MAP@3スコアの計算実装と検証

5. **モデル開発と改善**:
   - 多段階分類パイプラインの構築
   - 数学特化型モデルの実験
   - 交差検証戦略の設計
   - ハイパーパラメータチューニング

6. **提出準備**:
   - Notebookの実行時間最適化（9時間以内）
   - submission.csvの形式確認
   - リーダーボードでの検証

## リソース

- [Kaggle Competition Page](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)
- [Data Description](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/data)
- [Discussion Forum](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion)
- [Misconception Framework Report](リンクされたレポート)

## 引用

Jules King, Kennedy Smith, L Burleigh, Scott Crossley, Maggie Demkin, and Walter Reade. MAP - Charting Student Math Misunderstandings. https://kaggle.com/competitions/map-charting-student-math-misunderstandings, 2025. Kaggle.

## 重要な考察事項

### データの特性
- **OCR由来のテキスト**: 数学記号や式の表現に不一致がある可能性
- **学生の説明の多様性**: 同じ誤解でも表現方法が異なる
- **多段階の予測**: 正誤判定と誤解の特定を同時に行う必要性
- **教育的観点**: 誤解パターンの教育的意味を理解することの重要性

### 技術的チャレンジ
- **クラス不均衡**: 特定のカテゴリやMisconceptionが偏っている可能性
- **マルチラベル分類**: 最大3つの予測を適切に順位付け
- **ドメイン特化**: 一般的なNLPモデルでは数学的推論の理解が困難
- **評価指標の最適化**: MAP@3に特化した予測戦略の必要性

---

*最終更新: 2025年7月11日*