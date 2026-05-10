#ifndef CONE_CONES_H
#define CONE_CONES_H

/// @brief Base class for parameters that depend on a specific instance of a cone.
class ConeParameters{
protected:
    int barrier_parameter, num_variables;
public:
    virtual void parse_args(const std::string& args) = 0;
    int barrierParameter(){return barrier_parameter;}
    int numVariables(){return num_variables;}
};

/// @brief Base class for all cones.
class Cone{
private:
    Cone() = default; // By making the constructor private, the class cannot be instantiated.
protected:
    static const bool is_complex, is_symmetric;
};

/*
 * This way you can have an std::vector<std::string> that can be passed to a dispatcher for the Cones,
 * avoiding virtual functions, but you can also have an std::vector<std::unique_ptr<ConeParameters>>
 * list for the parameters of each cone. So you pass the parameter list for each call to any function.
*/

#endif