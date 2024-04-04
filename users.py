import configuration as cfg
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class User:
    id: int
    real_name: str
    background: str
    screen_name: str = None

def read_users_from_json_file(file_path:Path) -> List[User]:
    users = []
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        for user_id, user_info in json_data.items():
            user = User(
                id=user_info['id'],
                real_name=user_info['real_name'],
                background=user_info['background']
            )
            users.append(user)
    return users

USERS = read_users_from_json_file(cfg.JSON_USERS)

def get_user_by_id(user_id: int) -> User:
    for user in USERS:
        if user.id == user_id:
            return user

if __name__ == '__main__':
    print(USERS)