# -*- coding: utf-8 -*-
import datetime
import MySQLdb
from utility.data_reader import local_data_reader, db_to_dict
from ConvLSTM.config import db_config


def insert_to_db(db, insert_data):

    cursor = db.cursor()

    # insert data
    print("insert data .. ")
    for each_db_data in insert_data:
        cmd_str = "INSERT INTO ncsist_data(AMB_TEMP, PM2_5, RH, WIND_DIREC, WIND_SPEED, site, time) VALUES(%s, %s, %s, %s, %s, %s, %s)"
        val = (
            str(each_db_data['AMB_TEMP']), str(each_db_data['PM2.5']), str(each_db_data['RH']), str(each_db_data['WIND_DIREC']),
            str(each_db_data['WIND_SPEED']), str(each_db_data['site']), str(each_db_data['time'])
        )
        cursor.execute(cmd_str, val)

    # Make sure data is committed to the database
    db.commit()

    db.close()


def load_db(db, table_name, time_range=['2018-09-01', '2018-09-30']):

    cursor = db.cursor()

    # execute MySQL search command
    print("search data .. ")
    cmd_str = """SELECT * FROM %s where time >= "%s" and time <= "%s" order by time ASC""" % (
        table_name, time_range[0], time_range[1])
    cursor.execute(cmd_str)

    labels = [label[0] for label in cursor.description]
    # get all result of search
    results = cursor.fetchall()

    # close connection
    db.close()

    db_data = list()

    for each_data in results:
        db_data.append(dict())
        for label_idx, each_label in enumerate(labels):
            if each_label == "site" or each_label == "SiteName":
                db_data[-1][each_label] = each_data[label_idx]
            elif each_label == "time" or each_label == "PublishTime":
                db_data[-1][each_label] = datetime.datetime.strptime(each_data[label_idx], "%Y-%m-%d %H:%M:%S")
            else:
                db_data[-1][each_label] = float(each_data[label_idx])

    return db_data

# if __name__ == "__main__":
#
#     # ncsist_data_dir = "/media/clliao/006a3168-df49-4b0a-a874-891877a88870/AirQuality/dataset/PM25Data_forAI_0903-0930"
#     #
#     # ncsist_polution_db = local_data_reader(ncsist_data_dir)
#     # print("total data: %d" % len(ncsist_polution_db))
#
#     # connect MySQL
#     print("connect db .. ")
#     db = MySQLdb.connect(host=db_config["host"],
#                          user=db_config["user"], passwd=db_config["passwd"], db=db_config["db"])
#
#     # insert_to_db(db, insert_data=ncsist_polution_db)
#     ncsist_polution_db = load_db(db, table_name="AirDataTable")
#
#     ncsist_polution_data = db_to_dict(ncsist_polution_db)
#
#     exit()