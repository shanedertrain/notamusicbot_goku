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
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            for user_id, user_info in json_data.items():
                user = User(
                    id=user_info['id'],
                    real_name=user_info['real_name'],
                    background=user_info['background']
                )
                users.append(user)
    except FileNotFoundError:
        cfg.LOGGER.warning(f'File not found: {file_path}. This will be generated for your to fill in')
        return None
    return users

def get_user_by_id(user_id: int) -> User:
    for user in USERS:
        if user.id == user_id:
            return user

USERS = read_users_from_json_file(cfg.FILEPATH_USERS)

if __name__ == '__main__':
    print(USERS)