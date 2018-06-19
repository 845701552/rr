import os


class DelayPrint:
    def __init__(self):
        self.tasks = []
        self.will_print = True

    def inner_print(*args,file=None):
        print(*args)
        if file is not None:
            print(*args,file=file)


    def delay_print(self,*args):
        self.tasks.append(lambda file=None:DelayPrint.inner_print(*args,file=file))

    def add_print_task(self,task):
        self.tasks.append(task)

    def print_all(self,file_name=None):
        if self.will_print:
            if file_name is None:
                for task in self.tasks:
                    task()
            else:
                full_file_name = file_name+".txt"
                if not (os.path.isfile(full_file_name)):
                    with open(full_file_name, 'w'): pass
                with open(full_file_name, 'w') as file:
                    for task in self.tasks:
                        task(file=file)


    def print_all_and_flush(self):
        self.print_all()
        self.tasks = []
