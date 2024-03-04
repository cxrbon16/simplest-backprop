from math import log

class Item:
    def __init__(self, value: int = 0, grad: int = 0, children: list = []): 
        self.value = value
        self.grad = grad
        self.children = children
    
    def __repr__(self):
        return f"Item(value={self.value}, gradient={self.grad})"
    
    def __add__(self, other) -> 'Item':
        next = Item(value=self.value + other.value, children=[self,other])
        
        def backstep():
            self.grad = next.grad 
            other.grad = next.grad

        next._backstep = backstep
        return next
    
    def __mul__(self, other) -> 'Item':
        other = Item(other) if isinstance(other, int) else other

        next = Item(value=self.value * other.value, children=[self,other])
        
        def backstep():
            self.grad += next.grad * other.value
            other.grad += next.grad * self.value

        next._backstep = backstep
        return next
    
    def log(self, base: int = None) -> 'Item':
        value = log(self.value) if base is None else log(value, base)
        next = Item(value=value, children=[self])

        def backstep():
            self.grad += (1/self.value) * next.grad if base is None else (1/self.value) * log(base) * next.grad

        next._backstep = backstep
        return next
    

    def __sub__(self, other) -> 'Item':
        return self + (other * -1)
    
    def backprop(self): #bfs
        def order():
            visited = set()
            queue = [self]
            l = []
            while queue:
                parent = queue.pop(0)
                l.append(parent)
                for child in parent.children:
                    queue.append(child)
                visited.add(parent)
            return l
        
        self.grad = 1
        l = order()
        current = self
        while l:
            current = l.pop(0)
            if len(current.children) != 0:
                current._backstep()


        
    



a = Item(143)
b = Item(12)

c = a - b
c.grad = 12


c.backprop()

print(a.grad)