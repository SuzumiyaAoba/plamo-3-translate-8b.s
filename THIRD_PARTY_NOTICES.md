# 第三者のモデル・参考実装

このリポジトリの推論コード・Nix定義・テストは [MIT License](LICENSE) で公開します。モデル重み、語彙データ、公式Python実装は同梱していません。

## PLaMo 3 Translate

Built with PLaMo.

モデルは Preferred Networks の [pfnet/plamo-3-translate-8b](https://huggingface.co/pfnet/plamo-3-translate-8b) を利用します。固定リビジョンは `19ecab3fb31424ee937b7503730318b50e55771c` です。

追加モデルもPreferred Networksの公式配布を使用し、次のリビジョンに固定します。

| モデル | 公式配布元 | 固定リビジョン |
| --- | --- | --- |
| 0.8B | [pfnet/plamo-3-translate-0.8b](https://huggingface.co/pfnet/plamo-3-translate-0.8b) | `573eb1be9fd8c44402650810c680104e5aec81e8` |
| 2B | [pfnet/plamo-3-translate-2b](https://huggingface.co/pfnet/plamo-3-translate-2b) | `78355e2f59605d1c2c7c46e915333ffded24c8ff` |
| 31B | [pfnet/plamo-3-translate-31b](https://huggingface.co/pfnet/plamo-3-translate-31b) | `d6743d65c2cc61a07ff0fb2c3ff3cf7fd8ca36eb` |

利用者が公式配布元から別途取得する重み・語彙・付属ファイルには、公式の **PLaMo Community License** が適用されます。このリポジトリのMITライセンスは、モデルの利用条件を変更しません。取得・利用・出力の公開の前に、[日本語の公式ライセンス](https://huggingface.co/pfnet/plamo-3-translate-8b/blob/19ecab3fb31424ee937b7503730318b50e55771c/LICENSE/ja) または [英語版](https://huggingface.co/pfnet/plamo-3-translate-8b/blob/19ecab3fb31424ee937b7503730318b50e55771c/LICENSE/en) を確認してください。ダウンローダーは両言語のライセンスもモデル保存先へ取得します。

`tools/profiles/*/config.json`、`docs/config.json`、テンソルの名前・形状、RoPEの定数は、固定リビジョンの公開設定に基づく実装用メタデータです。学習済みの重みは含みません。`tools/profiles/*/manifest.json` と8B互換用の `tools/model-manifest.json` は配布元ファイルのサイズとSHA-256だけを記録しています。

公開設定のメタデータと、READMEの翻訳例・`docs/benchmarks` 内の翻訳テキストは、PLaMoに由来する情報・PLaMoモデルの出力です。これらの適用条件を参照できるよう、公式ライセンスのコピーを [日本語](LICENSES/PLaMo-Community.ja.txt)・[英語](LICENSES/PLaMo-Community.en.txt) で同梱しています。MITライセンスの対象と区別します。

## gpt2.s

手書きAArch64、Darwin syscall、元のモデル形式を直接読む設計は、Arnesh Banerjee の [gpt2.s](https://github.com/ArneshBanerjee/gpt2.s)（MIT License）を参考にしています。
