import mysql.connector
import os
from model_prediction import build_connect
os.environ["TORCH_HOME"] = "./torch"

# 实现查询功能
def retrival(table_name, search_query):
    cnx, cursor = build_connect()
    if not cnx or not cursor:
        print("Failed to connect to the database.")
        return
    try:
        search_query = f"%{search_query}%"
        retrival_query = f"select {table_name}.Text from {table_name} where {table_name}.Text LIKE %s"
        cursor.execute(retrival_query, (search_query,))
        results = cursor.fetchall()
        if results:
            print("Search Results:")
            for result in results:
                print(result)
        else:
            print("No matching records found! Please retrival again!")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        cnx.close()


table_name = 'CCB_marketing_story'


def main():
    search_query = input("请输入要检索的查询词：")
    retrival(table_name, search_query)


if __name__ == "__main__":
    main()
