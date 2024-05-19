import os

# Example Python string
my_string = """This is a test string.

轻度不佳
依照科学算命公司的测算，汝未来一段时日，或遇小挫。虽如春金般坚定果断，然在工作中或因过于追求效率，忽略细节，导致小失误。生活中亦或有小麻烦，如过度注重规则，令亲友感到拘束，偶有不和。

中度不佳
据科学算命公司之测算，汝未来一段时日，恐多有挑战。虽如春金般逻辑严谨，然在事业上或因过分强调秩序，导致团队士气低落，进展受阻。健康状况亦恐有起伏，时常感到压力和疲倦。人际关系方面亦或因过于严格，导致误解和摩擦，友朋之间生嫌隙，内心难得安宁。

严重不佳
依科学算命公司之测算，汝未来一段时日，恐遭遇严峻困境。虽如春金般坚定果断，然事业上或因过分追求目标，忽视团队需求，陷入困顿，难以进展，甚至有失败之虞。人际关系亦或因严苛要求，导致严重矛盾和冲突，感到孤立无援，内心焦虑不安，身心俱疲。

"""

# Convert the string to bytes
byte_data = my_string.encode('utf-8')

# Specify the path to the "123" folder on the Desktop
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
folder_path = os.path.join(desktop_path, '123')
file_path = os.path.join(folder_path, '123.bin')

print("code running here")

# Ensure the "123" folder exists
os.makedirs(folder_path, exist_ok=True)

# Open the file in binary write mode and write the bytes
with open(file_path, 'wb') as bin_file:
    bin_file.write(byte_data)


# # Specify the path to the "123" folder on the Desktop
# desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
# folder_path = os.path.join(desktop_path, '123')
# file_path = os.path.join(folder_path, 'output.txt')

# # Ensure the "123" folder exists
# os.makedirs(folder_path, exist_ok=True)

# # Open the file in text write mode and write the string
# with open(file_path, 'w', encoding='utf-8') as txt_file:
#     txt_file.write(my_string)

# print(f"String saved to {file_path}")

print(f"String saved to {file_path}")