import random as r

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution_twonumadd(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        yushu=0;
        index=0;
        l3=ListNode()
        l4 = l3
        while l1 is not None and l2 is not None:
            index+=1
            l3.val=(yushu+l1.val+l2.val)%10
            yushu=(yushu+l1.val+l2.val)/10
            l1=l1.next
            l2=l2.next
            l3.next=ListNode()
            l3=l3.next


        while l1 is not None:
            l3.val = (yushu + l1.val) % 10
            yushu=(yushu + l1.val) / 10
            l1=l1.next
            l3.next=ListNode()
            l3=l3.next

        while l2 is not None:
            l3.val = (yushu + l2.val) % 10
            yushu=(yushu + l2.val) / 10
            l2=l2.next
            l3.next=ListNode()
            l3=l3.next

        return l4














def main():
    tarnum=int(100*r.random())
    print(tarnum)
    iswin=False
    for i in range(10):
        num=int(input("请输入一个数："))
        if(num==tarnum):
            print("赢了！")
            iswin=True;
            break;
        elif(num>tarnum):
            print("大了")
        else:
            print("小了")

    if iswin==False:
        print("超过次数，游戏失败")
    # count=0;
    # for i in range(0,300,2):
    #     if i%2==0:
    #         if i%7==0 or i%17==0:
    #             print(i,end=" ")
    #             count+=1
    #             if count%5==0:
    #                 print()



if __name__ == '__main__':
    main()




