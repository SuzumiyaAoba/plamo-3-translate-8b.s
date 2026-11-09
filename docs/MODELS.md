# モデルの選択と検証範囲

0.8B・2B・8B・31Bの公式BF16チェックポイントに対応します。Nixの取得・CLI・APIは `--model SIZE` で選択でき、省略時は `PLAMO_MODEL` または8Bです。重みの量子化・学習・変換は行いません。

```sh
nix run .#download-model -- --model 2b --weights
nix run . -- --model 2b -j 'Hello.'
nix run .#serve -- --model 2b --memory auto
```

モデルごとに固定した[公式設定](../tools/profiles)から重みの形状、キャッシュ、作業領域、ファイルパスを生成します。演算カーネルは共通です。任意のHugging Faceモデルを動的に解釈する機能ではありません。`make models`とNixパッケージは、モデル本体を取得せず4種類のバイナリを生成します。

| モデル | 隠れ層の幅 | Attentionの幅 | 層数 | KVヘッド数 | BF16テンソル数 | ファイル数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 1536 | 2048 | 16 | 1 | 163 | 1 |
| 2B | 2560 | 3072 | 24 | 4 | 243 | 2 |
| 8B | 4096 | 4096 | 32 | 4 | 323 | 4 |
| 31B | 5120 | 8192 | 64 | 8 | 642 | 14 |

出力射影の形状は `[hidden_size, num_attention_heads * head_dim]` です。31Bの出力ヘッドは入力Embeddingと同じテンソルを参照します。全モデルの語彙、公式Python実装、基本プロンプト、終了トークン、ライセンスは固定したファイルのSHA-256が同一です。

## 確認した内容

- 全モデル: 合成BF16ファイルを実ローダーで読み込み、形状照合、部分失敗時の全解放、同一プロセスでの再試行、再初期化を検証。31Bは重み共有も確認。
- 0.8B・2B: 全重みのダウンロード時SHA-256、全テンソルの型・形状・アセンブリによる検索と登録を確認。
- 0.8B・2B: `Hello.`の英日翻訳が「こんにちは。」で終了。実装の先頭ID51871は公式FP32参照と一致。公式参照では続く終了IDは16で、翻訳全文も一致。
- 0.8B・2B: 常駐とストリームで入力処理後・生成後の全logitsと最終状態のハッシュが一致。各モードで連続2要求・キャンセル後の要求を確認。
- 0.8B・2B: 実HTTP APIで`I like programming.`の英日、`今日は良い天気です。`の日英が終了。同じワーカーの再利用とモデル識別子を確認。
- 8B: 既存の4つの基準ハッシュが一致し、従来の結果を維持。
- 31B: 公式14ファイルのヘッダーだけを取得し、全642テンソルをアセンブリで検索・形状照合・登録。31Bの実行ファイルで語彙読み込みと31ケースのトークン化も確認。

結果の要約は[検証記録](benchmarks/multi-model.json)にあります。検証機はApple Silicon・24 GiBです。31Bの重み本体の取得、全体での推論・翻訳・速度は未検証です。重みは63,514,989,488バイトあり、常駐の起動判定だけで61.15 GiBの再利用可能メモリが必要です。KV・入力処理・OS用の余裕も必要です。`stream`では1 GiBを超える行列をmmapから読みます。

これらは短い入力での互換性確認です。幅広い翻訳精度、公式BF16演算とのビット一致、長文・長いコンテキストの実モデル比較は含みません。8Bの既存結果の維持と、異なるモデル間での出力の一致は別の性質です。

## 再検証

```sh
# モデルなし。全形状のネイティブローダー、数値演算、HTTPのテスト
nix flake check

# 取得済みモデルの全ヘッダーとネイティブ検索・登録
python3 tools/verify_weights.py --model 0.8b
python3 tools/verify_weights.py --model 2b

# 31Bの重み本体を取得せず検証。ネット接続が必要
python3 tools/verify_weights.py --model 31b --headers-only --header-cache build/header-cache

# 常駐／ストリームを比較する記録を作成（必要な空きメモリを確保）
python3 tools/verify_api.py --model 2b --translate --memory resident --report build/resident.json
python3 tools/verify_api.py --model 2b --translate --memory stream --report build/stream.json

# PyTorch・Transformers等を別途導入した参照環境
.venv/bin/python tools/reference_first.py --model 2b --source en --target ja 'Hello.'
.venv/bin/python tools/reference_first.py --model 2b --source en --target ja 'Hello.' --append-id 51871
```

`--headers-only`はHTTP Rangeが無視された場合に読み込みを中止します。ヘッダーを一時的なスパースファイルに置いて本物の検索・登録処理を検証し、終了時に削除します。重み本体のSHA-256や推論結果を検証するものではありません。

Built with PLaMo. モデル・出力の利用条件は[第三者情報](../THIRD_PARTY_NOTICES.md)を参照してください。
