#ifndef CSVREADER_C
#define CSVREADER_C

/*----------------------------------------
  ***************************************
  |  read and store a CSV  |
  ***************************************
------------------------------------------*/
inline std::string GotoLine(std::fstream &file,int Nline) {
    file.seekg(std::ios::beg);
    for (int i=0;i<Nline-1;++i) {
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    std::string line;
    file >> line;
    return line;
};

/*----------------------------------------
  ***************************************
  |  read and store a CSV  |
  ***************************************
------------------------------------------*/
template <typename T>
inline void csvreader(std::string &line,std::vector<T> &data,int colstart,int colend,double scale,double shift) {
    std::string tmpline(line);
    tmpline.append(","); // Temporary fix for laziness

    //std::cout << tmpline << std::endl;

    data.clear();
    data.reserve(colend-colstart);

    int colpos = 0;
    while (tmpline.find_first_of(",")!=std::string::npos) {

        int tpos(tmpline.find_first_of(",")+1);

        if (colpos >= colstart && colpos <= colend)
        {
            data.push_back(atof(tmpline.substr(0,tpos-1).c_str())/(2.0*scale)+shift);
            if (data.back()>1.0 || data.back() < 0.0)
            {
                std::cout << "(" << colpos << "," << tpos << ") = " << tmpline.substr(0,tpos-1) << std::endl;
            }
        }

        tmpline=tmpline.substr(tpos);

        ++colpos;
    };
};

#endif
