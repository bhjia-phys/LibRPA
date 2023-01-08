//==========================================================
// AUTHOR : Peize Lin
// DATE : 2017-01-10
//==========================================================

#ifndef ABFS_VECTOR3_ORDER_H
#define ABFS_VECTOR3_ORDER_H
#include "vector3.h"
//#include "src_lcao/abfs.h"

//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/binary_iarchive.hpp>

template<typename T> class Vector3_Order: public Vector3<T>
{
public: 
	Vector3_Order(const Vector3<T> &v):Vector3<T>(v){}
	Vector3_Order(const T &x,const T &y,const T &z):Vector3<T>(x,y,z){}
	Vector3_Order()=default;
};

template<typename T>
bool operator< ( const Vector3_Order<T> &v1, const Vector3_Order<T> &v2 );

template<typename T>
Vector3_Order<T> operator-(const Vector3_Order<T> &v1)
{
    return {-v1.x, -v1.y, -v1.z};
}

template<typename T>
bool operator== ( const Vector3_Order<T> &v1, const Vector3_Order<T> &v2 );
/*
template<typename T>
bool operator> ( const Abfs::Vector3_Order<T> &v1, const Abfs::Vector3_Order<T> &v2 )
{
	if( v1.x>v2.x )       return true;
	else if ( v1.x<v2.x ) return false;
	if( v1.y>v2.y )       return true;
	else if ( v1.y<v2.y ) return false;
	if( v1.z>v2.z )       return true;
	else if ( v1.z<v2.z ) return false;
	return false;
}

template<typename T>
bool operator<= ( const Abfs::Vector3_Order<T> &v1, const Abfs::Vector3_Order<T> &v2 )
{
	if( v1.x<v2.x )       return true;
	else if ( v1.x>v2.x ) return false;
	if( v1.y<v2.y )       return true;
	else if ( v1.y>v2.y ) return false;
	if( v1.z<v2.z )       return true;
	else if ( v1.z>v2.z ) return false;
	return true;
}

template<typename T>
bool operator>= ( const Abfs::Vector3_Order<T> &v1, const Abfs::Vector3_Order<T> &v2 )
{
	if( v1.x>v2.x )       return true;
	else if ( v1.x<v2.x ) return false;
	if( v1.y>v2.y )       return true;
	else if ( v1.y<v2.y ) return false;
	if( v1.z>v2.z )       return true;
	else if ( v1.z<v2.z ) return false;
	return true;
}
*/

// (a%b+b)%b: return same sign % as b
template<typename T>
Vector3_Order<T> operator% ( const Vector3_Order<T> &v1, const Vector3_Order<T> &v2 )
{
	auto mod = [](const int i, const int n){ return (i%n+3*n/2)%n-n/2; };			// [-n/2,n/2]
//	auto mod = [](const int i, const int n){ return (i%n+n)%n; };					// [0,n]
	return Vector3_Order<T>{ mod(v1.x,v2.x), mod(v1.y,v2.y), mod(v1.z,v2.z) };
}

#endif	// ABFS_VECTOR3_ORDER_H
