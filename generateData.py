#! /usr/bin/env python
# coding : utf-8
import random

def generateWorkers(num, prob, skill_num, price_range):
    f = open("workerSet"+str(skill_num), 'w')
    for i in range(num):
        list_worker = []
        for j in range(skill_num):
            if random.random() < prob:
                list_worker.append("1")
            else:
                list_worker.append("0")
        price = random.randint(price_range[0], price_range[1])
        line = " ".join(list_worker) + "," + str(price) + "\n"
        f.write(line)
    f.close()


def generateTasks(num, prob, skill_num, price_range):
    f = open("TaskSet"+str(skill_num), 'w')
    for i in range(num):
        list_Task = []
        for j in range(skill_num):
            if random.random() < prob:
                list_Task.append("1")
            else:
                list_Task.append("0")
        price = random.randint(price_range[0], price_range[1])
        line = " ".join(list_Task) + "," + str(price) + "\n"
        f.write(line)
    f.close()


if __name__ == "__main__":
    generateWorkers(10, 0.4, 10, [10, 30])
    generateTasks(10, 0.6, 10, [100, 120])
    generateWorkers(20, 0.4, 20, [10, 30])
    generateTasks(10, 0.6, 20, [100, 120])
    generateWorkers(20, 0.4, 30, [10, 30])
    generateTasks(10, 0.6, 30, [100, 120])
