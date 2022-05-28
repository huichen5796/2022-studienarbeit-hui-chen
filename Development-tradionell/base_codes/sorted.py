from distutils.command.config import LANG_EXT


list1 = [(21,465),(23,565),(56,156),(78,552),(78,1),(15,22),(546,155),(45,12),(48,12),(847,121),(4545,125)]

list2 = sorted(list1, key = lambda x:x[0])

print(list2)
