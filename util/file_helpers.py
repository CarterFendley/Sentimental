import os

def gen_postfix(path):
    post_fix = 0
    while True:
        if post_fix == 0 and os.path.exists(path):
            post_fix +=1
        elif os.path.exists(f'{path}-{post_fix}'):
            post_fix +=1
        else:
            break
    if post_fix != 0:
        path += f'-{post_fix}'

    return path