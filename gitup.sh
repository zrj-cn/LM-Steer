echo "请输入更新说明："
read commit_message
commit_message=${commit_message:-"update"}

git add .
git commit -m "$commit_message"
git push origin main