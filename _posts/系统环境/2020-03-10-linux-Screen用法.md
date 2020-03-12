---
layout: post
comments: true
categories: 系统环境
---

Screen是一个可以在多个进程之间多路复用一个物理终端的全屏窗口管理器。Screen中有会话的概念，用户可以在一个会话中创建多个screen窗口，在每一个screen窗口中就像操作一个真实的telnet/SSH连接窗口那样。

通俗的讲，screen命令用于新建一个或多个“命令行窗口”，在新建的这“窗口”中，可以执行命令；每个“窗口”都是独立并行的。
### 1. 安装Screen
Ubuntu:         apt-get install screen
### 2. 使用Screen

(1). 创建会话：   screen -S xxx 
(2). 离开会话：   ctrl+a+d 
(3). 恢复会话：   screen -r xxx 
(4). 查看已经创建的会话：   screen -ls 
(5). 退出screen：   在screen下，输入 exit 
(6). 其他命令   
Ctrl + a，d ：   暂离当前会话 
Ctrl + a，c：   在当前screen会话中创建一个子会话 
Ctrl + a，w：   子会话列表 
Ctrl + a，p：   上一个子会话 
Ctrl + a，n ：   下一个子会话 
Ctrl + a，0-9 ：   在第0窗口至第9子会话间切换
### 3. 备注
在恢复screen时会出现There is no screen to be resumed matching ****，遇到这种情况咋办呢？输入命令
screen -d xxx
然后再使用恢复命令恢复

