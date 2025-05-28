#!/usr/bin/env bash
while getopts a:n:u:d: flag
do
    case "${flag}" in
        a) author=${OPTARG};;
        n) name=${OPTARG};;
        u) urlname=${OPTARG};;
        d) description=${OPTARG};;
    esac
done

echo "Author: $author";
echo "Project Name: $name";
echo "Project URL name: $urlname";
echo "Description: $description";

echo "Renaming project..."

original_author="jiahuigeng"
original_name="acl2025_con_instruction"
original_urlname="acl2025_con_instruction"
original_description="Awesome coninstruction created by jiahuigeng"
# Iterate over all files in the repository
git ls-files | while read -r filename; do
    # Exclude .github/workflows/rename_project.yml from renaming
    if [[ "$filename" != ".github/workflows/rename_project.yml" ]]; then
        sed -i "s/$original_author/$author/g" "$filename"
        sed -i "s/$original_name/$name/g" "$filename"
        sed -i "s/$original_urlname/$urlname/g" "$filename"
        sed -i "s/$original_description/$description/g" "$filename"
        echo "Renamed $filename"
    else
        echo "Skipping $filename"
    fi
done

mv "$original_name" "$name"

# This command runs only once on GHA!
rm -rf .github/template.yml
