now=$(date "+%Y-%m-%d %H:%M:%S")
echo "Change Directory to D:/计算机毕业设计"
cd D:/计算机毕业设计
echo "Starting add-commit-pull-push..."
git add . && git commit -m "$now" && git pull origin main && git push origin master:main
echo "Done!"