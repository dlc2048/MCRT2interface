import sys
from clang.cindex import Index, CursorKind


def generate_python_enum(header_file):
    index = Index.create()
    tu = index.parse(header_file)

    print("from enum import Enum, auto\n")

    for cursor in tu.cursor.walk_preorder():
        if cursor.kind == CursorKind.ENUM_DECL:
            enum_name = cursor.spelling
            print(f"class {enum_name}(Enum):")

            # Enum 멤버 순회
            for field in cursor.get_children():
                if field.kind == CursorKind.ENUM_CONSTANT_DECL:
                    val = field.enum_value
                    print(f"    {field.spelling} = {val}")
            print("")


if __name__ == "__main__":
    generate_python_enum(sys.argv[1])

