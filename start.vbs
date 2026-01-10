Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

' 切换到脚本所在目录
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir

' 检查虚拟环境是否存在
venvPath = scriptDir & "\venv"
If Not fso.FolderExists(venvPath) Then
    ' 首次运行：用 bat 脚本安装，显示详细日志
    WshShell.Run "cmd /k " & scriptDir & "\install.bat", 1, False
Else
    ' 正常启动：无窗口
    WshShell.Run "cmd /c call venv\Scripts\activate.bat && pythonw main.py", 0, False
End If
