#クラスの定義
class Man:
    def __init__(self, name): # コンストラクタの定義
        self.name = name # インスタンス変数nameに引数nameを代入
        print('Initialized!')
        
    def hello(self): # メソッドhelloの定義
        print('Hello ' + self.name + '!')

    def goodbye(self): # メソッドgoodbyeの定義
        print('Goodbye ' + self.name + '!')

m = Man('Jony')
m.hello()
m.goodbye()