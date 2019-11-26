#!/bin/bash


#WAYS TO LOOP OVER THE ELEMENTS OF AN ARRAY

array=('first' 'second' 'third' 'fourth') #array

#print the number of elements in the array
printf 'Number of elements in the array = %d \n\n' ${#array[*]}

#print the elements in the array
echo '###### array elements ######'
for element in ${array[*]}
do
    echo $element
done
echo '############################'

#print the elements in the array and their order
printf '\n###### array elements ######\n'
for i in ${!array[*]}
do
    printf 'element %d = %s\n' $i ${array[$i]}
done
printf '############################\n\n'

echo ${array[*]}   #print all the elements of the array
echo ${!array[*]}  #print the indexes of the array
echo ${#array[*]}  #print the number of elements in the array
