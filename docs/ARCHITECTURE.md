# コードの責務と変更時の確認

CLIと常駐ワーカーは同じエンジンを使います。各プロセスが1つのエンジンを所有し、要求は直列に処理します。

| 場所 | 責務 |
| --- | --- |
| `runtime/cli.c`・`runtime/worker.c` | 引数・プロセス間通信・結果の出力 |
| `runtime/engine.c` | モデルの準備、要求の状態、接頭KVの再利用、greedy生成と候補検証 |
| `runtime/tokenizer.c` | 語彙のマッピング、Trie、行索引、再利用する作業領域、UTF-8の符号化と復号 |
| `runtime/metal.m` | GPUの初期化・終了、共通バッファ、射影とMLP、CPUへのフォールバック |
| `runtime/metal_weights.m` | 重みの取得・保持・圧縮・解放、実行用ファイル、射影タイルの解決 |
| `runtime/metal_model.m` | KV・RoPE、モデル全体のprefill／decode、候補の検証と確定 |
| `runtime/parallel.c`・`src/` | CPUの実行スケジュール、AArch64／NEONの演算・I/O・トークナイザー |
| `tools/server.py` | HTTP、認証、入力検証、JSON／SSEの応答 |
| `tools/worker_client.py` | ワーカーの生存期間、要求の排他制御、キャンセル、UTF-8の再構成 |
| `tools/worker_protocol.py` | バイナリ要求と長さ制限付きJSONLイベント。APIと検証ツールで共有 |
| `tools/model_layout.py`・`tools/weight_schema.py`・`tools/rope_constants.py` | 固定プロファイルからアセンブリの記述子・重みスロット・定数を生成 |
| `Makefile`・`tests/Makefile` | モデル別の本体生成と、モデルなしで実行できる演算・境界テスト |

`runtime/model.h`と`runtime/compute.h`がCとアセンブリの接続を定義します。構造体のサイズ・オフセットはコンパイル時にも確認します。Metal内部の共有状態と補助関数は`metal_internal.h`に閉じ、呼び出し元には`metal.h`の操作だけを公開します。

## 保つ条件

- 元のBF16ビット、FP32の演算、CPUカーネルの加算順序を保つ。
- GPUのモデル実行は、状態やKVを書き換える前に全射影を解決する。解決失敗時はCPU経路へ移れる。
- GPUコマンドのエンコード中に重みを取得・解放しない。借りた領域はコマンドの完了まで有効にする。
- メモリ圧への対応、読み込み領域の上限、圧縮形式、モデルごとの形状・重み共有を保つ。
- トークナイザーが返すID・バイト列はそのインスタンスから借りる。次の操作より前に必要な内容を使うかコピーする。
- ワーカーのエラー応答は要求の失敗とプロセスの故障を区別する。キャンセル後は古い応答を読み終えてから次の要求を送る。

## 検証

```sh
make -j4 models
make test test-tools
nix flake check
```

Metalの検証には利用可能なGPU、HTTPの検証にはループバック通信が必要です。`SKIPPED`は実行済みの合格とは区別します。生成器の変更では4モデルの生成ファイルを変更前とバイト単位で比較し、`src/`の8B生成ファイルとの一致も確認します。

性能比較用バイナリは変更前に保存します。例えば0.8BのCPU比較は次のコマンドで実行できます。

```sh
python3 tools/benchmark_generation.py --model 0.8b \
  --baseline-worker /tmp/plamo-worker-before --worker build/0.8b/plamo-worker \
  --data-dir . --backend cpu --memory resident --storage raw \
  --repeats 5 --max-tokens 32 --report build/refactor-comparison.json
```

変更前後を交互に実行し、起動後・ウォームアップ後・測定後の保持する重み量と形式が同じ組だけを集計します。本文・生成ID・prefill後と生成後の全logits／状態ハッシュの一致も要求します。GPUの比較では`--backend metal`を使います。8Bの候補検証を測る場合は、両バイナリから対応する補助ワーカーと重みを参照できるようにします。
