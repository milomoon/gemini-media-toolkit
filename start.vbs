Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

' 切换到脚本所在目录
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' 检查虚拟环境是否存在
venvPath = scriptDir & "\venv"
If Not fso.FolderExists(venvPath) Then
    ' 首次运行：用 bat 脚本安装，显示详细日志
    WshShell.Run "cmd /k cd /d """ & scriptDir & """ && START.bat", 1, False
Else
    ' 正常启动：无窗口
    ' 使用 venv 里的 pythonw
    pythonwPath = scriptDir & "\venv\Scripts\pythonw.exe"
    mainPath = scriptDir & "\main.py"
    
    If fso.FileExists(pythonwPath) Then
        WshShell.Run """" & pythonwPath & """ """ & mainPath & """", 0, False
    Else
        ' venv 损坏，重新安装
        WshShell.Run "cmd /k cd /d """ & scriptDir & """ && START.bat", 1, False
    End If
End If
