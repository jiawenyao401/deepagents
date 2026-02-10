from deepagents.backends.filesystem import FilesystemBackend

fs_backend = FilesystemBackend(root_dir="D:\\test", virtual_mode=True)

print("=== Listing /skill/skills/skills ===")
items = fs_backend.ls_info("/skill/skills/skills")
for item in items:
    print(f"Path: {item['path']}, is_dir: {item.get('is_dir', False)}")
