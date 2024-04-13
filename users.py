import sys
import configuration as cfg
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List


@dataclass
class User:
    id: int
    screen_name: str
    model_name: str = "OBAMA"
    real_name: str = "Real name"
    background: str = "No background"

    def to_dict(self):
        result = asdict(self)
        return result

def read_users_from_json_file(file_path:Path=cfg.FILEPATH_USERS) -> List[User]:
    users = []
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            for user_id, user_info in json_data.items():
                user = User(
                    id=int(user_id),
                    real_name=user_info['real_name'],
                    background=user_info['background'],
                    model_name=user_info['model_name'],
                    screen_name=user_info['screen_name'],
                )
                users.append(user)
    except FileNotFoundError:
        cfg.LOGGER.warning(f'File not found: {file_path}. This will be generated for your to fill in')
        return None
    return users

def write_users_to_json(users:list[User], filename=cfg.FILEPATH_USERS):
    users_dict = [user.to_dict() for user in users]
    
    with open(filename, 'w') as file:
        json.dump(users_dict, file, indent=4)

def get_user_by_id(user_id: int) -> User:
    for user in USERS:
        if user.id == user_id:
            return user

USERS = read_users_from_json_file(cfg.FILEPATH_USERS)

if __name__ == '__main__':
    print(USERS)

    WRITE_FILEPATH = cfg.FOLDER_INPUT / 'users_test.json'
    write_users_to_json(USERS, WRITE_FILEPATH)
    print(f"Wrote users to file: {WRITE_FILEPATH}")

    USERS_NEW = read_users_from_json_file(cfg.FILEPATH_USERS)
    print(USERS_NEW)
    print(f"Read users from file: {WRITE_FILEPATH}")

    print(f"Users are the same: {USERS == USERS_NEW}")

    WRITE_FILEPATH.unlink()