import re

chat1 = "codebasics: you ask a lot of questions :) 12345678912, abc@xyz.com"
chat2 = "codebasics: here it is: (123)-567-8912, abc@xyz.com"
chat3 = "codebasics: yes, phone: 12345678912 email abc@xyz.com"

pattern1 = "\d{10}"

matches = re.findall(pattern1, chat1)

print(matches)

pattern2 = "\(\d{3}\)-\d{3}-\d{4}"

matches = re.findall(pattern2, chat2)

print(matches)

pattern3 = "\d{10}|\(\d{3}\)-\d{3}-\d{4}"

matches = re.findall(pattern3, chat3)

print(matches)

pattern_mail = "[a-z0-9A-Z_]*@[a-z0-9A-Z]*\.[a-zA-Z]*"

matches = re.findall(pattern_mail, chat3)

print(matches)

chat4 = "codebasics: Hello, I am having an issue with order # 1234567"

pattern4 = "order[^\d]*(\d*)"  # the round brackets are extracting only a substring

matches = re.findall(pattern4, chat4)

print(matches)
