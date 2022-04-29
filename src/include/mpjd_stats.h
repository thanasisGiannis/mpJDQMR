#pragma once


namespace mpjd{
  struct statistics{
      int m_matVecs     = 0; // number of matrix-vector (block matVec is assumed as block dimension matVecs)
      int m_numRestarts = 0; // number of basis restart
      int m_numOrth     = 0; // number of vectors orthogonalized
  };


  class mpjdStatistics {
      protected:
        static struct mpjd::statistics stats;
      
  };

  struct statistics mpjdStatistics::stats; // in order to be shared with the whole project
                                           // not sure if this is quite a good approach

  class matrixStatistics : public mpjdStatistics{
  
    public:
      void updateMatVecs(int counter){ (this->stats).m_matVecs+=counter;}
  
  };
  
  
  class jdStatistics : public mpjdStatistics{

  public:
    void printStats(){
            std::cout << "======================================" << std::endl;
            std::cout << "EigenSolver Statistics" << std::endl;
            std::cout << "======================================" << std::endl;
            
            std::cout << "#matVecs            = " << (this->stats).m_matVecs     << std::endl;
            std::cout << "#Restarts           = " << (this->stats).m_numRestarts << std::endl;
            std::cout << "#Orthogonilizations = " << (this->stats).m_numOrth     << std::endl;
            std::cout << "--------------------------------------" << std::endl;
        
        }

  };


  class basisStatistics : public mpjdStatistics{

  public:
    void updateRestarts(int counter){ (this->stats).m_numRestarts+=counter;}
    void updateOrthogonalizations(int counter){ (this->stats).m_numOrth+=counter;}

  };

}
