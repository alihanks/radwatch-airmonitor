#include <iostream>
#include <string>
#include <cstring>


char te[3]={'0','1','\0'};
char ts[3]={'3','4'};

int main(){


  char str[]="yes";
  for( int k=0; k<sizeof(str)-1; k++)
    std::cout << "," << str[k];

  std::cout << "\n" << sizeof(str);
  return 0;

}
