


class TreeNode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None


def create_default_tree():
    root = TreeNode(1)
    node1 = TreeNode(2)
    node2 = TreeNode(3)
    node3 = TreeNode(4)
    node4 = TreeNode(5)
    node5 = TreeNode(6)
    node6 = TreeNode(7)
    node7 = TreeNode(8)
    node8 = TreeNode(9)
    node9 = TreeNode(10)
    root.left = node1
    root.right = node2
    node1.left = node3
    node1.right = node4
    node2.left = node5
    node2.right = node6
    node3.left = node7
    node3.right = node8
    node4.left = node9
    return root


def getdfs(root):
    if root==None:
        return;
    else:
        print(root.value)
        getdfs(root.left)
        getdfs(root.right)



def main():
    root=create_default_tree()
    stack=[]
    stack.append(root)
    while(len(stack)!=0):
        tmproot=stack.pop()
        if tmproot!=None:
            print(tmproot.value)
            stack.append(tmproot.left)
            stack.append(tmproot.right)



if __name__ == '__main__':
    main()