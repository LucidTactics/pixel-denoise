' Double-click launcher for the Pixel Denoise GUI (no console window).
' Uses the local Anaconda pythonw; edit the path below if Python moves.
Dim sh, fso, dir, pyw
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = "C:\Users\Lucid\anaconda3\pythonw.exe"
If Not fso.FileExists(pyw) Then pyw = "pythonw.exe"   ' fall back to PATH
sh.CurrentDirectory = dir
sh.Run """" & pyw & """ """ & dir & "\denoise_gui.py""", 0, False
