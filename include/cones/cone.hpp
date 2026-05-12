#ifndef CONE_CONES_H
#define CONE_CONES_H

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