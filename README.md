# Setup notes

## Configuration files

Tell git to ignore changes in `config.py`

```
git update-index --assume-unchanged config.py
```

Add an API key to `config.py`

```
WANDB_API_KEY='dfd370b8341abb5f65701834fb331dd30827b9ba2c41f3417640289a6bf74f2e'
```

# Buggy things notes

- If mock data is changed, the fake cache will need to be regenerated, same for regular data