#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

//Helper method to change semicolons to commas in a given string
void applyComma(string & a) {
  for (int i = 0; i < a.length(); ++i)
    if (a[i]==';')
      a[i] = ',';
}

//Read each line from file and save each line into each element of vector <string>
vector <string> readFromFile(string directory, string csvName) {
  string csvDirectory = directory + csvName;
  ifstream fin(csvDirectory.c_str());
  //Initial vector <string> result
  vector <string> res;
  //Read until end of file
  while (!fin.eof()) {
    string s;
    //Store a whole line inside the temporary 's' variable
    getline(fin, s);
    //We need to store its directory
    s = directory + s;
    //Append into the vector string storeRows
    res.push_back(s);
  }
  fin.close();
  return res;
}

//write each line to the file corresponding with each element of the given vector <string>
void writeToFile(string directory, string csvName, vector <string> & storeRows) {
  string csvDirectory = directory + csvName;
  ofstream fout(csvDirectory.c_str());
  for (int i = 0; i < storeRows.size(); ++i)
    fout << storeRows[i] << endl;
  fout.close();
}

//Change all of semicolons into commas inside of the file with the given directory and filename
//And return their csv content
vector <string> semicolonToComma(string directory, string csvName) {
  //Each row of the file will be stored inside vector <string> storeRows
  vector <string> storeRows = readFromFile(directory, csvName);
  //Change all of semicolons inside the vector <string> into commas
  for (int i = 0; i < storeRows.size(); ++i)
    applyComma(storeRows[i]);
  //write to the same csv file again
  writeToFile(directory, csvName, storeRows);
  //return storeRows for later usage, but remove their first storeRows
  storeRows.erase(storeRows.begin());
  return storeRows;
}

//Return the image name of string s
string getImageName(string & s) {
  string res = "";
  for (int i = 0; i < s.length() && s[i] != ','; ++i)
    res.push_back(s[i]);
  return res;
}

string getClassId(string & s) {
  string res = "";
  for (int i = s.length()-1; i >= 0 && s[i] != ','; --i)
    res.push_back(s[i]);
  reverse(res.begin(), res.end());
  return res;
}

void createCSV(string csvName, vector <string> & storeRows) {
  ofstream fout(csvName.c_str());
  fout << "Directory,Class" << endl;
  //Iterate through each row of the content of csv
  for (int i = 0; i < storeRows.size(); ++i) {
    //Get the image name of the current row
    string imageName = getImageName(storeRows[i]);
    //Get the class id of the current row
    string classId = getClassId(storeRows[i]);
    fout << imageName << ',' << classId << endl;
  }
  fout.close();
}

void testSetInitial() {
  string directory = "testSet/GTSRB/Final_Test/Images/", csvName = "GT-final_test.csv";
  //Get the csv content
  vector <string> storeRows = semicolonToComma(directory, csvName);
  //Because we will be using pytorch dataloader, we need to initialize the csv file which consist of
  //the directory of each picture and their corresponding class id
  createCSV("test.csv", storeRows);
}

//example: 1 change to 00001 | 15 change to 00015
string intToClassId(int n) {
  string res = "";
  res.push_back((char)(n%10 + 48));
  n /= 10;
  res = (char)(n%10 + 48) + res;
  //////
  while (res.length() < 5)
    res = "0" + res;
  return res;
}

void trainSetInitial() {
  string directory = "trainSet/GTSRB/Final_Training/Images/";

  vector <string> storeRows;

  for (int i = 0; i < 43; ++i) {
    string classId = intToClassId(i);
    string csvName = "GT-"+classId+".csv";
    vector<string> current = semicolonToComma(directory+classId+"/", csvName);
    for (int i = 0; i < current.size(); ++i) 
      storeRows.push_back(current[i]);
  }

  createCSV("train.csv",storeRows);
}

int main() {
  testSetInitial();
  trainSetInitial();
  // scanSemicolonToComma('trainSet/GTSRB/Final_Trainning/Images/');
}
