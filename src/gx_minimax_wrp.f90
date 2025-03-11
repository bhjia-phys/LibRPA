! > \brief wrapper module to interface with greenX timefreqency library
module gx_minimax_wrp

  use iso_c_binding, only: c_int, c_double, c_ptr, c_f_pointer
  use gx_minimax, only: gx_minimax_grid, gx_minimax_grid_frequency
  use kinds, only: dp
  implicit none

  private
  public :: gx_minimax_grid_frequency_wrp, gx_minimax_grid_wrp

  ! internal allocatables for interfacing with greenX pure Fortran procedures
  real(dp), allocatable, dimension(:)   :: omega_points_f
  real(dp), allocatable, dimension(:)   :: omega_weights_f
  real(dp), allocatable, dimension(:)   :: tau_points_f
  real(dp), allocatable, dimension(:)   :: tau_weights_f
  real(dp), allocatable, dimension(:,:) :: cosft_wt_f
  real(dp), allocatable, dimension(:,:) :: cosft_tw_f
  real(dp), allocatable, dimension(:,:) :: sinft_wt_f

  contains

    subroutine allocate_freq_grids(num_points)
      integer, intent(in) :: num_points
      if (.not.allocated(omega_points_f)) allocate(omega_points_f(num_points))
      if (.not.allocated(omega_weights_f)) allocate(omega_weights_f(num_points))
    end subroutine

    subroutine deallocate_freq_grids()
      if (allocated(omega_points_f)) deallocate(omega_points_f)
      if (allocated(omega_weights_f)) deallocate(omega_weights_f)
    end subroutine

    subroutine allocate_time_grids(num_points)
      integer, intent(in) :: num_points
      if (.not.allocated(tau_points_f)) allocate(tau_points_f(num_points))
      if (.not.allocated(tau_weights_f)) allocate(tau_weights_f(num_points))
    end subroutine

    subroutine deallocate_time_grids()
      if (allocated(tau_points_f)) deallocate(tau_points_f)
      if (allocated(tau_weights_f)) deallocate(tau_weights_f)
    end subroutine

    subroutine allocate_trans_matrices(num_points)
      integer, intent(in) :: num_points
      if (.not.allocated(cosft_wt_f)) allocate(cosft_wt_f(num_points, num_points))
      if (.not.allocated(cosft_tw_f)) allocate(cosft_tw_f(num_points, num_points))
      if (.not.allocated(sinft_wt_f)) allocate(sinft_wt_f(num_points, num_points))
    end subroutine

    subroutine deallocate_trans_matrices()
      if (allocated(cosft_wt_f)) deallocate(cosft_wt_f)
      if (allocated(cosft_tw_f)) deallocate(cosft_tw_f)
      if (allocated(sinft_wt_f)) deallocate(sinft_wt_f)
    end subroutine

!> \brief Wrapper of gx_minimax_grid_frequency to obtain the minimax frequency grids
  subroutine gx_minimax_grid_frequency_wrp(num_points, e_min, e_max, omega_points, omega_weights, ierr) &
      &      bind(C, name="gx_minimax_grid_frequency_wrp")
    implicit none
  
    integer(kind=c_int), intent(in) :: num_points
    real(kind=c_double), intent(in) :: e_min, e_max
    type(c_ptr), value :: omega_points
    type(c_ptr), value :: omega_weights
    integer(kind=c_int), intent(out) :: ierr
  
    integer :: ig
  
    real(kind=c_double), pointer :: omega_points_fptr(:)
    real(kind=c_double), pointer :: omega_weights_fptr(:)
    call c_f_pointer(omega_points, omega_points_fptr, [num_points])
    call c_f_pointer(omega_weights, omega_weights_fptr, [num_points])

    call allocate_freq_grids(num_points) 
    call gx_minimax_grid_frequency(num_points, e_min, e_max, omega_points_f, omega_weights_f, ierr)
    do ig = 1, num_points
      omega_points_fptr(ig) = omega_points_f(ig)
      omega_weights_fptr(ig) = omega_weights_f(ig)
    enddo
    call deallocate_freq_grids()
  
  end subroutine 
  
!> \brief Wrapper of gx_minimax_grid to obtain the minimax time and frequency grids
  subroutine gx_minimax_grid_wrp(num_points, e_min, e_max, &
      &                          tau_points, tau_weights, omega_points, omega_weights, &
      &                          cosft_wt, cosft_tw, sinft_wt, &
      &                          max_errors, cosft_duality_error, ierr) &
      &      bind(C, name="gx_minimax_grid_wrp")
    implicit none
  
    integer(kind=c_int), intent(in) :: num_points
    real(kind=c_double), intent(in) :: e_min, e_max
    type(c_ptr), value :: omega_points, omega_weights, tau_points, tau_weights
    type(c_ptr), value :: cosft_wt, cosft_tw, sinft_wt
    type(c_ptr), value :: max_errors
    real(kind=c_double), intent(out) :: cosft_duality_error
    integer(kind=c_int), intent(out) :: ierr
  
    real(dp) :: max_errors_f(3)
    real(kind=c_double), pointer :: omega_points_fptr(:)
    real(kind=c_double), pointer :: omega_weights_fptr(:)
    real(kind=c_double), pointer :: tau_points_fptr(:)
    real(kind=c_double), pointer :: tau_weights_fptr(:)
    real(kind=c_double), pointer :: cosft_wt_fptr(:,:)
    real(kind=c_double), pointer :: cosft_tw_fptr(:,:)
    real(kind=c_double), pointer :: sinft_wt_fptr(:,:)
  
    integer :: ig, ig2
  
    call c_f_pointer(omega_points, omega_points_fptr, [num_points])
    call c_f_pointer(omega_weights, omega_weights_fptr, [num_points])
    call c_f_pointer(tau_points, tau_points_fptr, [num_points])
    call c_f_pointer(tau_weights, tau_weights_fptr, [num_points])
    call c_f_pointer(cosft_wt, cosft_wt_fptr, [num_points, num_points])
    call c_f_pointer(cosft_tw, cosft_tw_fptr, [num_points, num_points])
    call c_f_pointer(sinft_wt, sinft_wt_fptr, [num_points, num_points])
  
    call allocate_freq_grids(num_points) 
    call allocate_time_grids(num_points) 
    call allocate_trans_matrices(num_points)

    call gx_minimax_grid(num_points, e_min, e_max, tau_points_f, tau_weights_f, omega_points_f, omega_weights_f, &
      &                  cosft_wt_f, cosft_tw_f, sinft_wt_f, max_errors_f, cosft_duality_error, ierr)
  
    ! copy back the grids and weights
    omega_points_fptr = omega_points_f
    omega_weights_fptr = omega_weights_f
    tau_points_fptr = tau_points_f
    tau_weights_fptr = tau_weights_f
    ! matrix have to be transposed to account for different majors in Fortran and C/C++
    ! NOTE(minye): maybe it is better to do this in the C++ part?
    cosft_wt_fptr = transpose(cosft_wt_f)
    cosft_tw_fptr = transpose(cosft_tw_f)
    sinft_wt_fptr = transpose(sinft_wt_f)
  
    call deallocate_freq_grids() 
    call deallocate_time_grids() 
    call deallocate_trans_matrices()
  end subroutine
end module
