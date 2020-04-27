import csv
from csv import reader
from csv import writer
import pandas
df = pandas.DataFrame(data={"col1": ['list_1'], "col2": ['list_2']})
df.to_csv("./file.csv", sep=',',index=False)



# with open("./csv_file.csv","w+", newline="") as incsv:
#     wr = csv.DictWriter(incsv, fieldnames=["index", "comment", "sentiment"])
#     wr.writeheader()
#
# list_val = ["1", "Chào Chị Bảy, chào Hương Lê yêu yêu!", "0"]
# list_val2 = ["2","Mến chào anh tân thái kính chào quý bà con Ace trên diễn đàn yêu tự do yêu đất nước VN mình nhé",'0']
#
# # add new row to csv file using "a" mode
# with open("./csv_file.csv","a", encoding="utf8" ,newline="") as incsv:
#     wr = csv.writer(incsv, delimiter ="," )
#     wr.writerow([i for i in list_val])
#     wr.writerow([i for i in list_val2])
#
# #add a new column to csv file
# #copy to new file
# i=0
# new_col_val = ["prob", "0.01", "0.01", "0.01",]
# with open("./csv_file.csv", "r", encoding="utf8" ,newline="") as readcsv,\
#         open("./csv_out_file.csv", "w", encoding= "utf8" ,newline="") as writecsv:
#     csv_reader = reader(readcsv)
#     csv_write = writer(writecsv)
#     for row in csv_reader:
#         all = []
#         all.append(row)
#         all.append(new_col_val[i])
#         csv_write.writerow(all)
#         i+=1

# i=0
# new_col_val = ["prob", "0.01", "0.01", "0.01",]
# with open("./csv_out_file.csv", "a", encoding= "utf8" ,newline="") as outfile:
#        rd = reader(outfile)
#        wr = writer(outfile, delimiter =",")
#        for row in rd:
#            wr.writerow(row.append(new_col_val[i]))
#            i+=1



