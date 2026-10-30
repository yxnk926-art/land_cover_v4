"""Repository data-root ownership guards."""
from pathlib import Path


def assert_not_other_land_cover_repo(data_root, project_root):
    """Reject data roots whose real path belongs to another land_cover_* repo.

    External data mounts remain supported, but a repository-to-repository symlink
    is not: writing through such a link can silently mutate another archived
    experiment.  Both paths are resolved before the ancestry check.
    """
    root = Path(data_root).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    for ancestor in (root, *root.parents):
        if ancestor.name.startswith("land_cover_"):
            if ancestor != project:
                raise ValueError(
                    "data-root 的真实路径落在其他 land_cover_* 仓库内，拒绝读写："
                    f"{root}（当前仓库 {project}）"
                )
            break
    return root
