' Creates a Desktop shortcut to the fast Denoise launcher, with the tree icon.
Set ws = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
d = fso.GetParentFolderName(WScript.ScriptFullName)
desktop = ws.SpecialFolders("Desktop")
Set sc = ws.CreateShortcut(desktop & "\Pixel Denoise.lnk")
sc.TargetPath = "C:\Windows\System32\wscript.exe"
sc.Arguments = """" & d & "\Denoise.vbs"""
sc.WorkingDirectory = d
sc.IconLocation = d & "\icon.ico"
sc.Description = "AI sprite confetti/speck remover"
sc.Save
WScript.Echo "shortcut created on Desktop: " & desktop & "\Pixel Denoise.lnk"
