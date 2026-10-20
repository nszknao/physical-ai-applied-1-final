# 外部コード・モデル

## MicroDuckモデル

モデルの著作権者は **Pollen Robotics** です。

- 出典：[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
- 固定revision：`b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`
- 利用対象：`scene.xml`、`robot_groundcontact.xml`、これらが参照する38個のSTLメッシュ。
- 取得時の改変：なし。上流のファイル内容と相対配置を保持。
- MJCF（XML）：[Apache-2.0](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/LICENSE)。
- STLメッシュ：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。範囲・帰属表示の根拠は[上流のライセンス説明](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/LICENSE-MODELS.md)。

素材はGitリポジトリに同梱せず、`scripts/fetch_assets.py` で取得します。上流の `LICENSE` と `LICENSE-MODELS.md` も取得・保持します。取得対象とGit blobハッシュは `assets/sources.toml`、取得結果のSHA-256は `assets/vendor/microduck_rl/checksums.json` に記録します。

メッシュを共有・改変する際はPollen Roboticsへの帰属表示、出典とライセンスへのリンク、改変の表示、適用ライセンスの条件を保持してください。

## 実行ライブラリ

[MuJoCo](https://github.com/google-deepmind/mujoco)を物理シミュレーションと表示に使用します。Python依存は `pyproject.toml` で宣言します。
