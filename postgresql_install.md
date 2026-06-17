## Install postgresql local

```bash
sudo apt update
```

```bash
sudo apt install postgresql postgresql-contrib -y
```

```bash
sudo systemctl start postgresql
```

- **To auto start on boot**

```bash
sudo systemctl enable postgresql 
```

- **Switch to postgres Linux user**

```bash 
sudo -i -u postgres
```

- **Open psql shell**

```bash 
psql
```

- **Set password**

```bash 
ALTER USER postgres WITH PASSWORD 'postgres';
```
----

## Delete postgresql table

```bash 
DROP DATABASE <database_name>;
```

----

### Some Useful Commands

- **List all databases**
  
```bash 
\l
```

- **Connect to a database**

```bash 
\c <db_name>
```

- **List all tables in current DB**

```bash 
\dt
```

- **Describe a table (columns, types)**

```bash 
\d <tablename>
```

- **List all users/roles**

```bash 
\du
```

- **Quit psql**

```bash 
\q
```