# テキストの出典とライセンス / Text attribution & license

このリポジトリに含まれる **日本語の文章データ**（`corpus.json` の日本語パッセージ、
および `web/rounds.rinna.json` 内に表示用として保持される当該テキスト）は、
**日本語版ウィキペディア**（<https://ja.wikipedia.org/>）の記事から取得し、
抜粋・空白整形・冒頭の読み仮名カッコの除去などの **改変** を加えたものです。

- 原文ライセンス: **Creative Commons 表示-継承 4.0 国際（CC BY-SA 4.0）**
  <https://creativecommons.org/licenses/by-sa/4.0/deed.ja>
- 著作者: 各記事の執筆者（各記事の「履歴」ページを参照）
- 改変の有無: **あり**（抜粋・整形）

CC BY-SA 4.0 の **継承（ShareAlike）** 条項に従い、これら派生テキストも
同じ **CC BY-SA 4.0** の下で提供されます。テキストデータを再利用する場合は、
本ファイルの帰属表示を保持してください。

取得スクリプト: [`sample_wiki_ja.py`](sample_wiki_ja.py)

---

英語の文章データ（`corpus.en.json` / `web/rounds.gpt2.json` / `web/rounds.qwen3.json`）は
**OpenWebText**（<https://skylion007.github.io/OpenWebTextCorpus/>）に由来します。

注意: 本リポジトリの **ソースコード** のライセンスは上流プロジェクト
（<https://github.com/nickypro/gpt2-game>）に従います。上記の CC BY-SA 4.0 は
あくまで **テキストデータ部分** に対する帰属表示・ライセンスです。
