import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
import logging
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from simple_exact_diagonalization_routines.local_matrix_class import *
from math import isclose as equal_floats
class double_bracket_flow:
    def __init__( self, 
                    H, 
                    ## MAGNETIC: added attribute 'onsite_Z_coupling' for B, operator be set as B.dot(Z)
                    flow_generator = { 'type' : 'default', 'operator' : None , 'onsite_Z_coupling': None},
                    norm_type =  None,
                    flow_step_min = 0,
                    flow_step_max = 0.05,
                    nmb_search_points_minimizing_s_search = 100,
                    magnetic_step_min = 0,
                    magnetic_step_max = 1,
                    nmb_search_points_magnetic_b_search = 5,
                    nmb_flow_steps = 1,
                    custom_flow_step_list = None,
                    inverter = None,
                    trotter_steps = 5,
                    custom_magnetic_step_list = None,
                    please_use_binary_search = False, 
                    please_compute_flow_generator_norm = False,
                    please_use_group_commutator = False,
                    please_use_imperfect_group_commutator = False,
                    please_store_prescribed_generators = False,
                    please_be_verbose = False,
                    please_be_exhaustively_verbose = False,
                    please_be_exhaustively_visual = False,
                    please_be_visual = False  ):
        self.H =  H 
        self.L = int( np.log2( H.shape[0]) )

        self.flow_generator = flow_generator
        self.norm_type =  norm_type
        self.inverter = inverter
        self.trotter_steps = trotter_steps
        
        self.nmb_flow_steps = nmb_flow_steps

        self.flow_step_min = flow_step_min
        self.flow_step_max = flow_step_max
        self.nmb_search_points_minimizing_s_search = nmb_search_points_minimizing_s_search

        self.magnetic_step_min = magnetic_step_min
        self.magnetic_step_max = magnetic_step_max
        self.nmb_search_points_magnetic_b_search = nmb_search_points_magnetic_b_search

        self.please_use_binary_search = please_use_binary_search
        self.please_use_group_commutator = please_use_group_commutator
        self.please_use_imperfect_group_commutator = please_use_imperfect_group_commutator
        self.custom_flow_step_list = custom_flow_step_list
        self.custom_magnetic_step_list = custom_magnetic_step_list
        self.please_store_prescribed_flow_generators = please_store_prescribed_generators

        self.please_compute_flow_generator_norm = please_compute_flow_generator_norm
        self.please_compute_observables = False
        self.please_update_flow_unitary = False
        self.please_return_outputs = False
        self.please_be_verbose = please_be_verbose
        self.please_be_exhaustively_verbose = please_be_exhaustively_verbose
        self.please_be_visual = please_be_visual
        self.please_be_exhaustively_visual = please_be_exhaustively_visual
        self.please_evaluate_timing = False

        #It is not needed to initialize but just to summarize some naming conventions
        self.flow_outputs = {
        'flow_generator_type' : 'default',
        'minimal_norms_sigma_H' : [],
        'minimizing_flow_step' : [],
        'cumulative_flow_parameters' : [],
        'flow_steps_grids_binary_search' : [],
        'norms_flow_generator_W' : [],
        'flowed_hamiltonians' : [],
        'flowed_hamiltonian' : [],
        'config' : locals(),
        'norm_H_for_optimal_Z' : [],
        'norms_H_for_different_Z' : [],
        'all_norms_computed_in_search' : [],
        'all_Z_names' : [],
        'str_of_minimizer_Z' : [],
        }
        
        if self.please_be_exhaustively_verbose is True:
            logging.basicConfig(
                format="%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s",
                level=logging.INFO)
    @staticmethod
    def normalized(A):
        return A/np.linalg.norm(A)
    
    @staticmethod
    def commutator( A, B ):
            return A.dot( B ) - B.dot( A )

    @staticmethod
    def delta( A ):
        return np.diag( A.diagonal() )

    @staticmethod
    def sigma( A ):
        return A - double_bracket_flow.delta( A )

    def imperfect_inversion_evolution(self, t, N, H=None, H_inverter=None):
        # Returns approximated e^itH with trotterization
        if H is None:
            H = self.H
        if H_inverter is None:
            H_inverter = self.inverter
        H_imp_inv = H_inverter @ H @ H_inverter
        missing_term = H_imp_inv + H
        s = t/N
        return np.linalg.matrix_power(((H_inverter @ expm(-1j*s*H) @ H_inverter) @ expm(1j* s *missing_term)), N)
    
    def convergence(self, t, N, H=None, H_inverter=None):
        if H is None:
            H = self.H
        if H_inverter is None:
            H_inverter = self.inverter
        target = expm(-1j*t*H) @ self.imperfect_inversion_evolution(t, N, H, H_inverter)
        return np.abs(target.trace()/np.sqrt(np.size(H)))

    def choose_flow_generator( self, H = None ):
        # Check if H has been passed, if not, use self.H
        if H is None:
            H = self.H
            if self.please_be_exhaustively_verbose is True:
                print("Received None Hamiltonian in choose flow generator so setting to ||H|| = ", np.linalg.norm( H ) )
        else:
            if self.please_be_exhaustively_verbose is True:
                print("Received Hamiltonian in choose flow generator with ||H|| = ", np.linalg.norm( H ) )

        # flow_generator types: 'default'/'canonical', 'single_commutator', 
        # 'double_commutator', 'presribed'; does not check invalid types
        if self.flow_generator['type'] == 'default' or self.flow_generator['type'] == 'canonical':
            if self.please_be_exhaustively_verbose is True:
                print("Default generator computation... with ||H|| = ", np.linalg.norm( H ) )
            chosen_generator = double_bracket_flow.commutator( double_bracket_flow.delta( H ), double_bracket_flow.sigma( H )) 

        elif self.flow_generator['type'] == 'single_commutator':
            chosen_generator = double_bracket_flow.commutator( self.flow_generator['operator'], H ) 
            
        elif self.flow_generator['type'] == 'double_commutator':
            chosen_generator = 1j*double_bracket_flow.commutator( self.flow_generator['operator'], 
                                                          double_bracket_flow.commutator( self.flow_generator['operator'], H ) )  
        elif self.flow_generator['type'] == 'prescribed':
            chosen_generator =  self.flow_generator['operator']

        elif self.flow_generator['type'] == 'magnetic_field':
            Z = Z_class(self.L)
            Z_list = [Z.at(i+1) for i in range(self.L)]
            B_field = sum([double_bracket_flow.normalized(self.flow_generator['onsite_Z_coupling'])[i] * Z_list[i] for i in range (self.L)])
            chosen_generator = double_bracket_flow.commutator( B_field, H ) 
            
        if self.please_be_exhaustively_verbose is True:
            print("Chosen generator of type ", self.flow_generator['type'], " has norm ", np.linalg.norm( chosen_generator ) )
        return chosen_generator

    # Returns V=e^sW
    def unitary_flow_step( self, s, H = None, H_inverter = None, N = None):
        if H is None:
            H = self.H
        if H_inverter is None:
            H_inverter = self.inverter
        if N is None:
            N = self.trotter_steps
        sqrt_s = np.sqrt(s)
        D = self.choose_flow_generator( H )
        if self.please_use_group_commutator:
            return expm(-1j*sqrt_s*D) @ expm(-1j*sqrt_s*H) @ expm(1j*sqrt_s*D) @ expm(1j*sqrt_s*H)
        elif self.please_use_imperfect_group_commutator:
            invert_evolution = self.imperfect_inversion_evolution(sqrt_s, N, H, H_inverter)
            return expm(-1j*sqrt_s*D) @ expm(-1j*sqrt_s*H) @ expm(1j*sqrt_s*D) @ invert_evolution
        return  expm( s * self.choose_flow_generator( H ) )

    # Returns VHV^+ (rotated hamiltonian)
    def flow_step( self, s, H = None ):
        if H is None:
            H = self.H
            if self.please_be_exhaustively_verbose is True:
                print("Received None Hamiltonian in flow_step() so setting to ||H|| = ", np.linalg.norm( H ) )
        
        V = self.unitary_flow_step( s, H )  

        if self.please_update_flow_unitary is True:
            self.flow_unitary_V = self.flow_unitary_V.dot( V.T.conj() )
            if self.please_be_verbose is True:
                print( "Updated flow unitary V with step s = ",s)

        return V.dot( H.dot( V.T.conj() ) )
    
    def reflow_from_list( self, s_list, H = None, state = None, prescribed_generators = None, compute_observables_handle = None ):
        if H is None:
            H = self.H
        if self.please_be_verbose is True:
            print("In redo flow reseting the flow unitary" )
        self.please_update_flow_unitary = True
        self.flow_unitary_V = np.eye( 2**self.L )
        observables = []
        for i, s in enumerate(s_list):

            if self.please_be_verbose is True:
                print("Taking flow step ", i )

            if prescribed_generators is None:
                H = self.flow_step( s, H )
            # Given list of prescribed ops, set flow_gen = prescribed_generator[i].
            else:
                self.flow_generator['type'] = 'prescribed'
                self.flow_generator['operator'] = prescribed_generators[ i ]

            if compute_observables_handle is None:
                self.compute_observables( H = H, state = state )
            else:
                observables.append( compute_observables_handle( H ) )

        return observables

    def compute_expected_value( self, state = None, observable = None ):
        if observable is None:
            observable = self.H
        if state is None:
            state = self.initial_state
        return state.conj().T.dot( observable ).dot( state )

    def compute_expected_energy( self, state = None, H = None ):
        expected_energy = self.compute_expected_value( state, H )
        if self.please_be_exhaustively_verbose is True:
            print( "<H> = ", expected_energy )
        return expected_energy

    # Returns <H^2> - <H>^2
    def compute_expected_energy_fluctuation( self, state = None, H = None, expected_energy  = None ):
        if H is None:
            H = self.H

        if expected_energy is None:
            expected_energy = self.compute_expected_energy( state, H )
        energy_fluctuation = H.dot( H ) - expected_energy**2 * np.eye(2**self.L)  

        expected_energy_fluctuation = self.compute_expected_value( state, energy_fluctuation )
        if self.please_be_exhaustively_verbose is True:
            print( "<H^2> - <H>^2 = ", expected_energy_fluctuation )
        return expected_energy_fluctuation
    
    # Update computational state and energy/fluc
    def compute_observables( self, state = None, H = None ):
        if state is None:
            if self.please_update_flow_unitary is True:
                state = self.flow_unitary_V.dot( self.state )
            else:
                state = self.initial_state

        expected_energy = self.compute_expected_energy( state, H ) 
        expected_energy_fluctuation = self.compute_expected_energy_fluctuation( state, H, expected_energy = expected_energy ) 
        self.store_flow_output( expected_energy = expected_energy, expected_energy_fluctuation = expected_energy_fluctuation )

    def compute_observables_during_flow( self, initial_state = None ):
        if initial_state is None:
            self.do_not_compute_observables_during_flow()
            return 0
        self.state = initial_state
        self.flow_unitary_V = np.eye( 2**self.L )
        self.please_compute_observables = True
        self.please_update_flow_unitary = True

    def do_not_compute_observables_during_flow( self ):
        self.flow_unitary_V = None
        self.please_compute_observables = False
        self.please_update_flow_unitary = False

    def find_minimizing_flow_step( self, H = None ): 
        if self.please_evaluate_timing is True:
            start = time.clock()
        if H is None:
            if self.please_be_exhaustively_verbose is True:
                print( "Received None H in find minimizing flow step so setting to ||H|| = ", np.linalg.norm(H) )
            H = self.H
        else:
            if self.please_be_exhaustively_verbose is True:
                print( "Received some H in find minimizing flow step with ||H|| = ", np.linalg.norm(H) )
            
        norms_sigma_H_s = []

        whether_to_update_flow_unitary = self.please_update_flow_unitary
        self.please_update_flow_unitary = False

        # For non-binary search, go through all s-grid sequentially and find min
        if self.please_use_binary_search is False:

            if self.custom_flow_step_list is None:
                s_grid = np.linspace( self.flow_step_min, self.flow_step_max, self.nmb_search_points_minimizing_s_search )
                # print('s_grid', s_grid)
            else:
                s_grid = self.custom_flow_step_list
            
            for s in s_grid:       
                if self.please_be_exhaustively_verbose is True:
                    print("Searching for minimizing flow step using s = ", s, " and flow generator type ", self.flow_generator['type'] )
                    print("Going to pass s = ", s, " ||H|| = ", np.linalg.norm( H ) )
                H_s = self.flow_step( s, H )
                norms_sigma_H_s.append( np.linalg.norm( double_bracket_flow.sigma( H_s ), self.norm_type ) )
            
            minimal_norm_sigma_H_s = np.min( norms_sigma_H_s )
            minimizing_flow_step = s_grid[ np.argmin( norms_sigma_H_s )]
            optimally_flowed_H = self.flow_step( minimizing_flow_step, H )         

        else:
            s_grid = []
            s_R = self.flow_step_max 
            s_L = self.flow_step_min

            H_R = self.flow_step( s_R, H )
            norm_R = np.linalg.norm( double_bracket_flow.sigma( H_R ), self.norm_type )
            norms_sigma_H_s.append( norm_R )

            H_L = self.flow_step( s_L, H )
            norm_L = np.linalg.norm( double_bracket_flow.sigma( H_L ), self.norm_type )
            norms_sigma_H_s.append( norm_L )

            s_grid.append( s_R )
            s_grid.append( s_L )
            for i in range(self.nmb_search_points_minimizing_s_search):
                s_mid = s_L + ( s_R - s_L ) / 2
                H_mid = self.flow_step( s_mid, H )
                norm_mid = np.linalg.norm( double_bracket_flow.sigma( H_mid ), self.norm_type )
                norms_sigma_H_s.append( norm_mid )
                s_grid.append( s_mid )
                if self.please_be_exhaustively_verbose is True:
                    print(i, s_L, s_mid, s_R, norm_L, norm_mid, norm_R)

                if norm_R < norm_L:
                    s_L = s_mid
                    norm_L = norm_mid
                else:
                    s_R = s_mid
                    norm_R = norm_mid
            minimal_norm_sigma_H_s = np.min( norm_mid )
            minimizing_flow_step = s_mid
            optimally_flowed_H = H_mid    
            

        if self.please_be_verbose is True:
            if self.please_evaluate_timing is True:
                print( "    Found minimum norm = ", minimal_norm_sigma_H_s, " for s = ", minimizing_flow_step, " in ", time.clock() - start, " s" )
            else:
                print( "    Found minimum norm = ", minimal_norm_sigma_H_s, " for s = ", minimizing_flow_step )

            
        if self.please_be_visual is True:
            plt.plot( s_grid, norms_sigma_H_s )
            plt.grid()
            if self.save_path is not None:
                plt.save( self.save_path, format = 'pdf' )
            plt.show()
        
        self.please_update_flow_unitary = whether_to_update_flow_unitary

        return {'optimally_flowed_H' : optimally_flowed_H, 'minimizing_flow_step' : minimizing_flow_step,
       'minimal_norm_sigma_H_s' : minimal_norm_sigma_H_s, 's_grid' : s_grid, 'all_norms_computed_in_search' : norms_sigma_H_s }

    def store_flow_outputs_for_plotting( self, optimally_flowed_H, minimizing_flow_step, minimal_norm_sigma_H_s, s_grid,
    norms_sigma_H_s, **other_outputs):
        self.store_flow_output( 'minimizing_flow_step', minimizing_flow_step )
        self.store_flow_output( 'minimal_norm_sigma_H', minimal_norm_sigma_H_s )
        self.store_flow_output( 's_grids', s_grid )
        self.store_flow_output( 'all_norms_computed_in_search', norms_sigma_H_s )
        self.store_flow_output( 'cumulative_flow_parameters', sum( self.flow_outputs['minimizing_flow_steps'] ) )
        if self.please_be_verbose is True:
            print("Flow step output with generator type ", self.flow_generator['type'], " gave minmal norm ", 
              minimal_norm_sigma_H_s, " for s = ", minimizing_flow_step )

    def store_flow_output( self, **outputs ):
        for output_key in outputs:
            if output_key in self.flow_outputs:
                self.flow_outputs[ output_key ].append( outputs[ output_key ] )
            else:
                self.flow_outputs[ output_key ] = [  outputs[ output_key ] ]

    def store_initial_H_in_outputs_for_plotting( self, H ):
        self.store_flow_output(   optimally_flowed_H  = self.H, 
                                                minimizing_flow_step = 0, 
                                                minimal_norm_sigma_H_s = np.linalg.norm( double_bracket_flow.sigma( H ) ),
                                                s_grid = [0], 
                                                norms_sigma_H_s = [] )

    def flow_forwards( self, H = None, states = None ):
        # If H not given, continue from what we have
        if H is None:
            H = self.H
            if 'final_flowed_H' in self.flow_outputs:
                H = self.flow_outputs['final_flowed_H'][0]
                if self.please_be_verbose is True:
                    print("In flow forwards continuing with the previous flowed H")

        whether_to_compute_observables = self.please_compute_observables
        if states is not None:
            self.please_compute_observables = True
        
        self.store_initial_H_in_outputs_for_plotting( H )
        self.store_flow_output( norms_flow_generator_W = np.linalg.norm( self.choose_flow_generator( H ) ) )
        
        if self.please_be_exhaustively_verbose is True:
            print( self.flow_outputs ) 

        for i in range( self.nmb_flow_steps ):
            if self.please_be_verbose is True:
                print( "Flow step ", i, " using H with norm ", np.linalg.norm( H ) )
            output_flow_step = self.find_minimizing_flow_step( H )
            
            H = output_flow_step[ 'optimally_flowed_H' ]
            if self.please_be_exhaustively_verbose is True:
                print( output_flow_step )
            if equal_floats( output_flow_step['minimizing_flow_step'], 0 ):
                break

            self.store_flow_output( **output_flow_step )
            self.store_flow_output(  norms_flow_generator_W = np.linalg.norm( self.choose_flow_generator( H ) ) )

            if self.please_store_prescribed_flow_generators is True:
                self.store_flow_output( 'prescribed_flow_generators',  self.choose_flow_generator( H ) )
            
            if self.please_compute_observables is True:
                if states is None:
                    self.compute_observables( H = H )
                else:
                    for state in states:
                        self.compute_observables( H = H, state = np.array( state ) )
        
        self.please_compute_observables = whether_to_compute_observables

        self.store_flow_output( final_flowed_H = H )

        if self.please_return_outputs is True:
            return self.flow_outputs

    def flow_via_local_Z_sweep( self, H = None, states = None, Z = None, use_single_commutator = True ):
        if H is None:
            if 'final_flowed_H' in self.flow_outputs:
                H = self.flow_outputs['final_flowed_H'][0]
                if self.please_be_verbose is True:
                    print("In flow via local sweep continuing with the previous flowed H")
            else:
                H = self.H
                if self.please_be_verbose is True:
                    print("In flow via local sweep using initial H")
        if Z is None:
            Z = Z_class(self.L)

        self.store_initial_H_in_outputs_for_plotting( H )
        self.store_flow_output( norms_flow_generator_W = np.linalg.norm( self.choose_flow_generator( H ) ) )
        
        if self.please_be_exhaustively_verbose is True:
            print( self.flow_outputs )
        if use_single_commutator is True:
            self.flow_generator['type'] = 'single_commutator'
        else:
            self.flow_generator['type'] = 'double_commutator'

        for i in range( self.nmb_flow_steps ):
            for x in range( self.L ):
                if self.please_be_verbose is not False:
                    print( "Flow step ", i, " using H with norm ", np.linalg.norm( H ) )
                    print( "Sweeping with Z_", x ) 

                self.flow_generator['operator'] = Z.at(x+1)    
                output_flow_step = self.find_minimizing_flow_step( H )

                H = output_flow_step[ 'optimally_flowed_H' ]
                if self.please_be_exhaustively_verbose is True:
                    print( output_flow_step )

                self.store_flow_output( **output_flow_step )
                self.store_flow_output(  norms_flow_generator_W = np.linalg.norm( self.choose_flow_generator( H ) ) )

                if self.please_store_prescribed_flow_generators is True:
                    self.store_flow_output( 'prescribed_flow_generators',  self.choose_flow_generator( H ) )
                
                if self.please_compute_observables is True:
                    if states is None:
                        self.compute_observables( H = H )
                    else:
                        for state in states:
                            self.compute_observables( H = H, state = np.array( state ) )

                self.store_flow_output( final_flowed_H = H )

                if self.please_return_outputs is True:
                    return self.flow_outputs


    def flow_via_selected_Z_search( self, H = None, states = None, Z = None, 
   please_also_check_canonical_bracket = True, please_use_single_commutator = True, Z_words = []  ):
   
        if H is None:
            if 'final_flowed_H' in self.flow_outputs:
                H = self.flow_outputs['final_flowed_H'][0]
                if self.please_be_verbose is True:
                    print("In flow via selected Z continuing with the previous flowed H") 
            else:
                H = self.H
                if self.please_be_verbose is True:
                    print("In flow via selected Z flowing with initial H") 
        if Z is None:
            Z = Z_class(self.L)
            
        #Prepare iteration
        self.store_initial_H_in_outputs_for_plotting( H )
        self.store_flow_output( norms_flow_generator_W = np.linalg.norm( self.choose_flow_generator( H ) ) )
        
        if self.please_be_exhaustively_verbose is True:
            print( "Starting flow via selected Zs", self.flow_outputs )
            print( "Flow step ", i, " using H with off-diagonal norm", np.linalg.norm( self.sigma(H) ) )      

        
        #Begin the iteration loop
        for i in range( self.nmb_flow_steps ):

            if self.please_be_verbose is True:
                print( "Flow step ", i )
                
            #Search for optimal direction
            norms_H_for_different_Z = []
            if Z_words is not []:
               for word in Z_words:
                   
                   #Set direction        
                   if please_use_single_commutator is True:
                       self.flow_generator['type'] = 'single_commutator'
                   else:
                       self.flow_generator['type'] = 'double_commutator'
                   self.flow_generator['operator'], str_name = Z.word( word )
                   if self.please_be_exhaustively_verbose is True:
                       print(  self.flow_generator['operator'].diagonal() )
                   #Find optimal step
                   output_flow_step = self.find_minimizing_flow_step( H )
                   minimal_norm_H_found = output_flow_step['minimal_norm_sigma_H_s']
                   norms_H_for_different_Z.append( minimal_norm_H_found )

                   if self.please_be_verbose is True:
                       print( "Flow step output with generator type ", self.flow_generator['type'], "using string ", str_name, " gave for s = ", output_flow_step['minimizing_flow_step'], "  minimal norm ", minimal_norm_H_found )

                   #Store outputs
                   if i == 0:
                       self.store_flow_output( all_Z_names = str_name )
        
            other_Z_ops = []
            other_Z_ops_names = []

            other_Z_ops.append( np.sum(  Z.at(x+1) for x in range(0,self.L+1,2)  ) )
            other_Z_ops_names.append( 'Magnetization_odd' )

            other_Z_ops.append( np.sum(  Z.at(x+1) for x in range(1,self.L,2)  ) )
            other_Z_ops_names.append( 'Magnetization_even' )

            other_Z_ops.append( np.sum(  np.cos( np.pi*x/self.L) * Z.at(x+1) for x in range(self.L)  ) )
            other_Z_ops_names.append( 'Magnetization_cos' )

            other_Z_ops.append( self.delta( H ) )
            other_Z_ops_names.append( 'Diagonal_H' )
            
            switch_odd = np.eye(2**self.L)
            #Works only in odd dimensions....
            if self.please_be_exhaustively_verbose is True:
                print([ x for x in range(0,self.L+1,2) ])
                print([ int(x) for x in range(0,self.L+1,2) ])
            for x in range(0,self.L+1,2):
                if self.please_be_exhaustively_verbose is True:
                    print(x)
                switch_odd = switch_odd.dot( Z.at(x+1) )
            other_Z_ops.append( switch_odd )
            other_Z_ops_names.append( 'switch_odd' )

            switch_even = np.eye(2**self.L)
            for x in range(1,self.L,2):
                switch_even = switch_even.dot( Z.at(x) )
            other_Z_ops.append( switch_even )
            other_Z_ops_names.append( 'switch_even' )
            if please_also_check_canonical_bracket is True:
               other_Z_ops.append( self.delta( H ) )
               other_Z_ops_names.append( 'Canonical' )
               
            norms_other_Z_ops = []
            for Z_op, name in zip( other_Z_ops, other_Z_ops_names ):
                
                #Set direction        
                if please_use_single_commutator is True:
                    self.flow_generator['type'] = 'single_commutator'
                else:
                    self.flow_generator['type'] = 'double_commutator'

                self.flow_generator['operator'] = Z_op
                
                if self.please_be_exhaustively_verbose is True:
                    print(  self.flow_generator['operator'].diagonal() )

                #Find optimal step
                output_flow_step = self.find_minimizing_flow_step( H )
                minimal_norm_H_found = output_flow_step['minimal_norm_sigma_H_s']
                norms_other_Z_ops.append( minimal_norm_H_found )

                if self.please_be_verbose is True:
                    print( "Flow step output with generator type ", self.flow_generator['type'], "using string ", name, " gave for s = ", output_flow_step['minimizing_flow_step'], "  minimal norm ", minimal_norm_H_found )

                #Store outputs
                if i == 0:
                    self.store_flow_output( all_Z_names = name )
  
            if min( norms_other_Z_ops ) < min( norms_H_for_different_Z ):
               norms_other_Z_ops_is_better = True
            else:
               norms_other_Z_ops_is_better = False    
          
              
            if norms_other_Z_ops_is_better is False:
                #Find which direction is best
                norm_H_minimal = np.min( norms_other_Z_ops )
                ind_H_min = np.argmin( norms_other_Z_ops )
                
                #Store which direction is best
                self.store_flow_output( strs_of_minimizer_Z = other_Z_ops_names[ind_H_min] )

                #Recompute the flow for the best direction
                self.flow_generator['type'] = 'single_commutator'
                self.flow_generator['operator'] = other_Z_ops[ind_H_min] 
                if self.please_be_verbose is True:
                  print( other_Z_ops_names[ind_H_min], " was found to be the best operator")
            
            else:
                #Store which direction is best
                norm_H_minimal = np.min( norms_H_for_different_Z )
                ind_H_min = np.argmin( norms_H_for_different_Z )

                #Recompute the flow for the best direction
                self.flow_generator['type'] = 'single_commutator'
                self.flow_generator['operator'], a = Z.word( self.flow_outputs['all_Z_names'][ind_H_min] )
                if self.please_be_verbose is True:
                  print( self.flow_outputs['all_Z_names'][ind_H_min], " was found to be the best operator")
            
            #Better store the minimizing s's because it's a small data structure while one finding can take minutes for largesystem sizes....

            
            if self.please_be_verbose is True:
                print('Flowing forwards with type ', self.flow_generator['type'])
            output_flow_step = self.find_minimizing_flow_step( H )
            H = output_flow_step[ 'optimally_flowed_H' ]

            self.store_flow_output( **output_flow_step )
            self.store_flow_output(  norms_flow_generator_W = np.linalg.norm( self.choose_flow_generator( H ) ) )
            
            if self.please_store_prescribed_flow_generators is True:
                self.store_flow_output( prescribed_flow_generators =  self.choose_flow_generator( H ) )
            
            if self.please_compute_observables is True:
                if states is None:
                    self.compute_observables( H = H )
                else:
                    for state in states:
                        self.compute_observables( H = H, state = np.array( state ) )

            if self.please_be_exhaustively_verbose is True:
                print( output_flow_step )
                print(i, " so for so good: ", self.flow_outputs['minimal_norm_sigma_H_s'][-1], norm_H_minimal,
                      ind_H_min,
                      self.flow_outputs['all_Z_names'][ind_H_min], np.linalg.norm(double_bracket_flow.sigma(H)) )
            if self.please_be_verbose is True:
                print(self.flow_outputs['str_of_minimizer_Z'])
        
        self.store_flow_output( final_flowed_H = H )
        
        if self.please_return_outputs is True:
            return self.flow_outputs

    def flow_via_best_Z_search( self, H = None, states = None, Z = None, 
   please_also_check_canonical_bracket = True, please_use_single_commutator = True  ):
        if H is None:
            if 'final_flowed_H' in self.flow_outputs:
                H = self.flow_outputs['final_flowed_H'][0]
            else:
                H = self.H
        if Z is None:
            Z = Z_class(self.L)
        #Prepare iteration
        self.store_initial_H_in_outputs_for_plotting( H )
        self.store_flow_output( norms_flow_generator_W = np.linalg.norm( self.choose_flow_generator( H ) ) )
        
        if self.please_be_exhaustively_verbose is True:
            print( self.flow_outputs )
        
        for i in range( self.nmb_flow_steps ):

            if self.please_be_verbose is True:
                print( "Flow step ", i )
            if self.please_be_exhaustively_verbose is True:
                print( "Flow step ", i, " using H with off-diagonal norm", np.linalg.norm( self.sigma(H) ) )      

            #Search for optimal direction
            norms_H_for_different_Z = []
            for idx in itertools.product(*[ ['0','1'] for s in range( self.L )]):
                
                #Set direction        
                if please_use_single_commutator is True:
                    self.flow_generator['type'] = 'single_commutator'
                else:
                    self.flow_generator['type'] = 'double_commutator'
                self.flow_generator['operator'], str_name = Z.word( idx )
                if self.please_be_exhaustively_verbose is True:
                    print(  self.flow_generator['operator'].diagonal() )
                #Find optimal step
                output_flow_step = self.find_minimizing_flow_step( H )
                minimal_norm_H_found = output_flow_step['minimal_norm_sigma_H_s']
                norms_H_for_different_Z.append( minimal_norm_H_found )

                if self.please_be_verbose is True:
                    print( "Flow step output with generator type ", self.flow_generator['type'], "using string ", str_name, " gave for s = ", output_flow_step['minimizing_flow_step'], "  minimal norm ", minimal_norm_H_found )

                #Store outputs
                if i == 0:
                    self.store_flow_output( all_Z_names = str_name )
            if please_also_check_canonical_bracket is True:
                self.flow_generator['type'] = 'canonical'
                output_flow_step = self.find_minimizing_flow_step( H )
                minimal_norm_H_found = output_flow_step['minimal_norm_sigma_H_s']

                if minimal_norm_H_found < min( norms_H_for_different_Z ):
                    if self.please_be_verbose is True:
                        print("Found that canonical is better because ",minimal_norm_H_found , min( norms_H_for_different_Z) )
                    canonical_bracket_better = True
                else:
                    if self.please_be_verbose is True:
                        print("Found that canonical is worse because ",minimal_norm_H_found , min( norms_H_for_different_Z) )
                    canonical_bracket_better = False
                norms_H_for_different_Z.append( minimal_norm_H_found )
            else:
                canonical_bracket_better = False
            
            #Store what is the best performance of individual directions when varying s
            self.store_flow_output( norms_H_for_different_Z = norms_H_for_different_Z )
            
            if canonical_bracket_better is False:
                #Find which direction is best
                norm_H_minimal = np.min( norms_H_for_different_Z )
                ind_H_min = np.argmin( norms_H_for_different_Z )
                
                #Store which direction is best
                self.store_flow_output( strs_of_minimizer_Z = self.flow_outputs['all_Z_names'][ind_H_min] )

                #Recompute the flow for the best direction
                self.flow_generator['type'] = 'single_commutator'
                self.flow_generator['operator'], a = Z.word( self.flow_outputs['all_Z_names'][ind_H_min] )
            else:
                #Store which direction is best
                norm_H_minimal = np.min( norms_H_for_different_Z )
                ind_H_min = np.argmin( norms_H_for_different_Z )
                self.store_flow_output( strs_of_minimizer_Z = 'Canonical' )

                #Recompute the flow for the best direction
                self.flow_generator['type'] = 'canonical'
            
            output_flow_step = self.find_minimizing_flow_step( H )
            
            if self.please_be_verbose is True:
                print('Flowing forwards with type ', self.flow_generator['type'])
            H = output_flow_step[ 'optimally_flowed_H' ]

            self.store_flow_output( **output_flow_step )
            self.store_flow_output(  norms_flow_generator_W = np.linalg.norm( self.choose_flow_generator( H ) ) )
            if self.please_store_prescribed_flow_generators is True:
                self.store_flow_output( 'prescribed_flow_generators',  self.choose_flow_generator( H ) )
            
            if self.please_compute_observables is True:
                if states is None:
                    self.compute_observables( H = H )
                else:
                    for state in states:
                        self.compute_observables( H = H, state = np.array( state ) )

            if self.please_be_exhaustively_verbose is True:
                print( output_flow_step )
                print(i, " so for so good: ", self.flow_outputs['minimal_norm_sigma_H_s'][-1], norm_H_minimal,
                      ind_H_min,
                      self.flow_outputs['all_Z_names'][ind_H_min], np.linalg.norm(double_bracket_flow.sigma(H)) )
            if self.please_be_verbose is True:
                print(self.flow_outputs['str_of_minimizer_Z'])
        
        self.store_flow_output( final_flowed_H = H )
        
        if self.please_return_outputs is True:
            return self.flow_outputs

    def find_onsite_Z_coupling_gradient( self, s, H = None, B = None, dB = 0.001):
        if H is None:
            H = self.H
        # Initial B field
        if B is None:
            B = self.flow_generator['onsite_Z_coupling']
            B_original = np.copy(B).astype(float)
        else:
            B_original = np.copy(self.flow_generator['onsite_Z_coupling']).astype(float)
            self.flow_generator['onsite_Z_coupling'] = B
        derivative_B = []
        # ||sigma(H_B0)||
        H_B0 = self.flow_step(s, H)
        norms_sigma_H_B0 = np.linalg.norm( double_bracket_flow.sigma( H_B0 ), self.norm_type)
        # Find derivative for each entry
        B0 = np.copy(B).astype(float)
        for i in range (self.L):
            # ||sigma(H_B1||
            B = np.copy(B0)
            B[i] += dB
            self.flow_generator['onsite_Z_coupling'] = np.copy(B)
            H_B1 = self.flow_step(s,H)
            norms_sigma_H_B1 = np.linalg.norm( double_bracket_flow.sigma( H_B1 ), self.norm_type )
            derivative_B.append((norms_sigma_H_B1 - norms_sigma_H_B0)/dB)
        
        self.flow_generator['onsite_Z_coupling'] = np.copy(B_original)
        return double_bracket_flow.normalized(derivative_B)
   
    def flow_via_onsite_Z_potential_search(self, H = None, states = None, B = None, 
   please_also_check_canonical_bracket = True, please_use_single_commutator = True):
        if H is None:
            if 'final_flowed_H' in self.flow_outputs:
                H = self.flow_outputs['final_flowed_H'][0]
            else:
                H = self.H
        if B is None:
            B = double_bracket_flow.normalized(self.flow_generator['onsite_Z_coupling'])
            self.flow_generator['onsite_Z_coupling'] = B
        else:
            self.flow_generator['onsite_Z_coupling'] = double_bracket_flow.normalized(B)
            B = self.flow_generator['onsite_Z_coupling']
            
        #Prepare iteration
        self.store_initial_H_in_outputs_for_plotting( H )
        self.store_flow_output( norms_flow_generator_W = np.linalg.norm( self.choose_flow_generator( H ) ) )
        
        if self.please_be_exhaustively_verbose is True:
            print( self.flow_outputs )
        
        if self.custom_magnetic_step_list is None:
                b_grid = np.linspace( self.magnetic_step_min, self.magnetic_step_max, self.nmb_search_points_magnetic_b_search )
        else:
            b_grid = self.custom_magnetic_step_list
        
        for i in range( self.nmb_flow_steps + 1 ):
            if self.please_be_verbose is True:
                    print( "Flow step ", i, "using H with norm", np.linalg.norm( H ) )
            if self.please_be_exhaustively_verbose is True:
                print( "Flow step ", i, " using H with off-diagonal norm", np.linalg.norm( self.sigma(H) ) )
            norms_H_for_different_db = []
            searched_B_directions = []
                # The first step
            if i == 0:
                # TODO Option for search in list, now assume first step direction B given
                output_flow_step = self.find_minimizing_flow_step( H )
                minimal_norm_H_found = output_flow_step['minimal_norm_sigma_H_s']

            else:      
                # Gradient descend direction
                B_descend = double_bracket_flow.normalized(self.find_onsite_Z_coupling_gradient(output_flow_step['minimizing_flow_step'],H))
                # print('Flowing with gradient', B_descend)
                # TODO If B_descend = 0?
                B0 = np.copy(B)
                # Find best direction with respectivve best sigma descrease
                verbose = self.please_be_verbose
                for db in b_grid:
                    if self.please_be_exhaustively_verbose:
                        print(' db =',db, ', B =', B - B_descend * db)
                    else:
                        self.please_be_verbose = False
                    new_B = double_bracket_flow.normalized(B - B_descend * db)
                    searched_B_directions.append(new_B)
                    self.flow_generator['onsite_Z_coupling'] = new_B
                    output_flow_step = self.find_minimizing_flow_step( H )
                    minimal_norm_H_found = output_flow_step['minimal_norm_sigma_H_s']
                    norms_H_for_different_db.append(minimal_norm_H_found)
                    B = np.copy(B0)
                self.please_be_verbose = verbose
                
                self.store_flow_output( searched_B_directions = searched_B_directions)   
                self.store_flow_output( norms_H_for_different_db = norms_H_for_different_db)
                    
                # Save best direction so far
                ind_H_min = np.argmin( norms_H_for_different_db )
                best_B = double_bracket_flow.normalized(B - B_descend * b_grid[ind_H_min])
                if please_also_check_canonical_bracket is True:
                    if self.please_be_verbose:
                        self.flow_generator['type'] = 'canonical'
                    if self.please_be_exhaustively_verbose:
                        print(' Canonical')
                    else:
                        self.please_be_verbose = False
                    output_flow_step = self.find_minimizing_flow_step( H )
                    minimal_norm_H_found = output_flow_step['minimal_norm_sigma_H_s']
                    self.please_be_verbose = verbose

                    if minimal_norm_H_found < min( norms_H_for_different_db ):
                        # if self.please_be_verbose is True:
                        #     print("Found that canonical is better because canonical",minimal_norm_H_found, ", magnetic", min( norms_H_for_different_db) )
                        canonical_bracket_better = True
                    else:
                        # if self.please_be_verbose is True:
                        #     print("Found that canonical is worse, because canonical",minimal_norm_H_found , ", magnetic", min( norms_H_for_different_db) )
                        canonical_bracket_better = False
                    norms_H_for_different_db.append( minimal_norm_H_found )
                else:
                    canonical_bracket_better = False    
                
                if canonical_bracket_better is False:
                    #Find which direction is best
                    ind_H_min = np.argmin( norms_H_for_different_db )
                    self.store_flow_output( list_of_minimizer_B = self.flow_outputs['searched_B_directions'][0][ind_H_min])
                    
                    #Recompute the flow for the best direction
                    self.flow_generator['type'] = 'magnetic_field'
                    self.flow_generator['onsite_Z_coupling'] = self.flow_outputs['searched_B_directions'][0][ind_H_min]
                else:
                    #Store which direction is best
                    self.store_flow_output( list_of_minimizer_B = 'Canonical' )
                    #Recompute the flow for the best direction
                    self.flow_generator['type'] = 'canonical'
                if self.please_be_exhaustively_verbose:
                    print(" Recalculate with best", 'canonical' if canonical_bracket_better else 'magnetic field')
                
                # Recalculate
                if self.please_be_exhaustively_verbose == False:
                    self.please_be_verbose = False
                output_flow_step = self.find_minimizing_flow_step( H )
                self.please_be_verbose = verbose
                H = output_flow_step[ 'optimally_flowed_H' ]
                self.store_flow_output( **output_flow_step)
                self.store_flow_output( norms_flow_generator_W = np.linalg.norm(self.choose_flow_generator(H)))
                
                if self.please_be_verbose is True:
                    if canonical_bracket_better:
                        print('For step', i, ', the minimum norm found', minimal_norm_H_found, 'with canonical, step size s = ', output_flow_step['minimizing_flow_step'])
                    else:
                        print('For step', i, ', the minimum norm found', min( norms_H_for_different_db ), 'with magnetic field B = ', self.flow_generator['onsite_Z_coupling'], 'step size s = ', output_flow_step['minimizing_flow_step'])
                    
                # Get back to magnetic search if canonical was better
                self.flow_generator['type'] = 'magnetic_field'
                self.flow_generator['onsite_Z_coupling'] = best_B
        self.store_flow_output( final_flowed_H = H )
        
        if self.please_return_outputs is True:
            return self.flow_outputs

    def flow_ini_state( self, ini_state, H, outputs = None ):
        
        if self.store_a_lot is not None:
            psis_flow = [ini_state]
        energy = []
        energy_fluct = []
        norms = []
        V = np.eye(2**self.L) 
        self.flow_generator = {}
        self.flow_generator['type'] = 'single_double_bracket_flow.commutator'
        observables = []
        observable = Z.at(1)
        for i in range( self.nmb_flow_steps ):
           
            for x in range(self.L):
                self.flow_generator['operator'] = Z.at(x+1)    
                if 1:#outputs is None:
                    V = V.dot( expm( minimizing_s*double_bracket_flow.commutator(Z.at(x+1),H) ) )
                psi = (V.dot( psi_ini ))
                norms.append(np.linalg.norm(double_bracket_flow.sigma(H)))

                energy.append( (psi.conj().T.dot(H).dot(psi)))
                energy_fluct.append( 
                    psi.conj().T.dot((H - energy[-1]).dot(H - energy[-1])).dot(psi)) 
                echoes = []
                for t in np.linspace(.1,5,10):
                    psi_noneq = expm(-1j*t*H_TFIM).dot(psi)
                    echoes.append( abs(psi_noneq.conj().T.dot(psi) ))
                observables.append(echoes)    
                H = H_flow
            norms.append(np.linalg.norm(double_bracket_flow.sigma(H)))
            print(np.linalg.norm(double_bracket_flow.sigma(H)))
        return energy, energy_fluct, norms, observables, V

    ### Plot flow evaluation: $\double_bracket_flow.sigma$ decrease, flow step sizesfrom mpl_toolkits.axes_grid1 import make_axes_locatable
     
    def show_flowed_H( self, H = None, flow_results = None, save_path = None ):
        if flow_results is None:
            flow_results = self.flow_outputs
        if H is None:
            H = flow_results['final_flowed_H'][-1]
        L = self.L 
        f=plt.figure(figsize = ( 20,4 ))
#a        
        ax_a = f.add_subplot(1,1,1)

#plot data
        plt.imshow(H)
        plt.colorbar()
        plt.xlabel(r'i') 
        plt.xlabel(r'j')
        title_str = r'Flowed H, L=' +str(self.nmb_flow_steps)
        plt.title( title_str )

        #panel label
        a =-.1
        b = 1.05
        plt.annotate('a)', xy = (a,b), xycoords='axes fraction')


    def show_flowed_observable( self, observable_name, show_subset = None, flow_results = None, save_path = None, please_use_trivial_x_axis = False ):
        if flow_results is None:
            flow_results = self.flow_outputs
        L = self.L

        f=plt.figure(figsize = ( 20,4 ))
        if show_subset is not None:
            k_range = show_subset
        else:
            k_range = range( 2**L )
        for k in k_range:
            obs = self.observable_over_steps( k, flow_results[ observable_name ] )
            plt.plot( obs )
        plt.show()
        

    def show_flow_outputs_norm( self, flow_results = None, save_path = None, please_use_trivial_x_axis = False ):
        if flow_results is None:
            flow_results = self.flow_outputs
        L = self.L 
        f=plt.figure(figsize = ( 20,4 ))
#a        
        ax_a = f.add_subplot(1,3,1)

#plot data
        norms = flow_results['minimal_norm_sigma_H_s']
        if please_use_trivial_x_axis is True:
            x_axis = np.linalg(1,len(norms)+1,len(norms) )
        else:
            x_axis = [sum(flow_results['minimizing_flow_step'][:k])for k in range( 1, len(flow_results['minimizing_flow_step'])+1)]

#plot
        plt.plot( x_axis, norms, '-o')   
        plt.grid()
        
        plt.xlabel(r'Flow duration $s$')
        plt.title(r'Norm off-diagonal $\vert\vert\sigma(H_k)\vert\vert$')

        #panel label
        a =-.1
        b = 1.05
        plt.annotate('a)', xy = (a,b), xycoords='axes fraction')
        plt.show()


    def show_flow_forwards_results( self, flow_results = None, save_path = None, please_save_the_fig = False):
        if flow_results is None:
            flow_results = self.flow_outputs
        L = self.L 
        f=plt.figure(figsize = ( 20,4 ))
#a        
        ax_a = f.add_subplot(1,3,1)

#plot data
        norms = flow_results['minimal_norm_sigma_H_s']
        x_axis = [sum(flow_results['minimizing_flow_step'][:k]) for k in range( 1, len(flow_results['minimizing_flow_step'])+1)]
#plot
        plt.plot( x_axis, norms, '-o')   

       
#axis labels
        x_labels_rounded = [ round(x, 2 ) for x in x_axis ]
        x_labels_rounded = [0] + x_labels_rounded[0:5] + [max(x_labels_rounded)]
        x_labels_rounded.pop(3)
        plt.xticks(x_labels_rounded)

        y_labels_rounded = [ round(y, 1 ) for y in norms ]
        y_labels_rounded = y_labels_rounded[0:5] + [min(y_labels_rounded)]
        plt.yticks(y_labels_rounded)

        plt.grid()
        
        plt.xlabel(r'Flow duration $s$')
        plt.title(r'Norm off-diagonal $\vert\vert\sigma(H_k)\vert\vert$')

        #panel label
        a =-.1
        b = 1.05
        plt.annotate('a)', xy = (a,b), xycoords='axes fraction')

#b
        f.add_subplot(1,3,2)
        plt.annotate('b)', xy = (a,b), xycoords='axes fraction')

        ini_norm = flow_results['minimal_norm_sigma_H_s'][0]
        plt.plot(flow_results['s_grid'][1],flow_results['all_norms_computed_in_search'][0]-ini_norm,'-', label=r'First step $k=1$')
        plt.plot(flow_results['s_grid'][2],flow_results['all_norms_computed_in_search'][1]-ini_norm,'-', label=r'Second step $k=2$')
        x_labels_rounded = [0]+[ round(x, 2 ) for x in flow_results['minimizing_flow_step'][1:3] ]
        y_labels_rounded = [ round(y-ini_norm, 1 ) for y in flow_results['minimal_norm_sigma_H_s'][0:2] ]
        y_labels_rounded.insert(0,0)
        plt.xticks(x_labels_rounded)
        plt.yticks(y_labels_rounded)
        plt.grid()

        plt.xlabel(r'Flow step $s$')
        plt.title(r'Norm change $\vert\vert\sigma(H_{k}))\vert\vert-\vert\vert\sigma(H)\vert\vert$')
        plt.legend(loc='best', bbox_to_anchor=(0.8,0.25))
#c
        ax = f.add_subplot(1,3,3)
        plt.annotate('c)', xy = (a,b),
                     xycoords='axes fraction')

        #inset the last Hamiltonian
        plt.imshow(flow_results['final_flowed_H'][-1].real, cmap='RdBu')

        plt.colorbar()
        if save_path is None:
            save_path = 'figs/flow_forwards_N_steps_'+str(self.nmb_flow_steps)+'_L_'+str(self.L)+'.pdf'
        if please_save_the_fig is True:
            plt.savefig( save_path, format='pdf')
        plt.show()
        
    def show_flow_forwards_results_fancy( self, flow_results = None, save_path = None ):
        if flow_results is None:
            flow_results = self.flow_outputs
        L = self.L 
        f=plt.figure(figsize = ( 20,4 ))
#a        
        ax_a = f.add_subplot(1,3,1)

#plot data
        norms = flow_results['minimal_norm_sigma_H_s']
        x_axis = [sum(flow_results['minimizing_flow_step'][:k])for k in range( 1, len(flow_results['minimizing_flow_step'])+1)]

#plot
        plt.plot( x_axis, norms, '-o')   

        #annotation steps
        k=0
        for xy in zip([x+.041 for x in x_axis[0:4]], 
                      [n + 0.05 for n in norms[0:4]]):                                       # <--
            if k == 0:
                plt.annotate('Initial norm', xy=(.05,norms[0]-.2), 
                             textcoords='data')
            elif k == 1:
                plt.annotate('Flow step\n           k = %d' % k, xy=xy, textcoords='data')
            else:
                plt.annotate('k = %d' % k, xy=xy, textcoords='data')
            k=k+1

#axis labels
        x_labels_rounded = [ round(x, 2 ) for x in x_axis ]
        x_labels_rounded = [0] + x_labels_rounded[0:5] + [max(x_labels_rounded)]
        x_labels_rounded.pop(3)
        plt.xticks(x_labels_rounded)

        y_labels_rounded = [ round(y, 1 ) for y in norms ]
        y_labels_rounded = y_labels_rounded[0:5] + [min(y_labels_rounded)]
        plt.yticks(y_labels_rounded)

        plt.grid()
        
        plt.xlabel(r'Flow duration $s$')
        plt.title(r'Norm off-diagonal $\vert\vert\sigma(H_k)\vert\vert$')

        #panel label
        a =-.1
        b = 1.05
        plt.annotate('a)', xy = (a,b), xycoords='axes fraction')

#inset initial Hamiltonian
        axin1 = ax_a.inset_axes([0.06,.12, 0.31, 0.31])

        inset1 = axin1.imshow( self.H, cmap='RdBu' )

        divider1 = make_axes_locatable(axin1)
        div_ax = divider1.append_axes("right", size="10%", pad=0.05)
        cbar1 = plt.colorbar(inset1, cax = div_ax )
        
        axin1.set_yticks(range(2**L))  
        axin1.set_xticklabels([1]+['']*(2**L-2)+[2**L])
        axin1.set_yticklabels([1]+['']*(2**L-2)+[2**L])
        axin1.set_xticks(range(2**L))
        ax_a.annotate(' ',
                xy=(x_axis[0], norms[0]),  
                xytext=(0.085, 0.39),    
                textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='bottom')

#inset the last Hamiltonian
        axin2 = ax_a.inset_axes([0.62,.63, 0.31, 0.31])
        inset2 = axin2.imshow(flow_results['final_flowed_H'][-1], cmap='RdBu')

        divider2 = make_axes_locatable(axin2)
        div_ax2 = divider2.append_axes("right", size="10%", pad=0.05)
        cbar2 = plt.colorbar(inset1, cax = div_ax2 )
        ax_a.annotate(' ',
            xy=(x_axis[-1], norms[-1]), 
            xytext=(0.85, 0.55),  
            textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left',
            verticalalignment='bottom')

        axin2.set_yticks(range(2**L))  
        axin2.set_xticklabels([1]+['']*(2**L-2)+[2**L])
        axin2.set_yticklabels([1]+['']*(2**L-2)+[2**L])
        axin2.set_xticks(range(2**L))
#b
        f.add_subplot(1,3,2)
        plt.annotate('b)', xy = (a,b), xycoords='axes fraction')

        ini_norm = flow_results['minimal_norm_sigma_H_s'][0]
        plt.plot(flow_results['s_grid'][1],flow_results['all_norms_computed_in_search'][0]-ini_norm,'-', label=r'First step $k=1$')
        plt.plot(flow_results['s_grid'][2],flow_results['all_norms_computed_in_search'][1]-ini_norm,'-', label=r'Second step $k=2$')
        x_labels_rounded = [0]+[ round(x, 2 ) for x in flow_results['minimizing_flow_step'][1:3] ]
        y_labels_rounded = [ round(y-ini_norm, 1 ) for y in flow_results['minimal_norm_sigma_H_s'][0:2] ]
        y_labels_rounded.insert(0,0)
        plt.xticks(x_labels_rounded)
        plt.yticks(y_labels_rounded)
        plt.grid()

        plt.xlabel(r'Flow step $s$')
        plt.title(r'Norm change $\vert\vert\sigma(H_{k}))\vert\vert-\vert\vert\sigma(H)\vert\vert$')
        plt.legend(loc='best', bbox_to_anchor=(0.8,0.25))
#c
        ax = f.add_subplot(1,3,3)
        plt.annotate('c)', xy = (a,b),
                     xycoords='axes fraction')
        ini_norm = flow_results['minimal_norm_sigma_H_s'][0]
        k=11
        plt.plot(flow_results['s_grid'][-1],flow_results['all_norms_computed_in_search'][k],'-', 
                 label=r'Intermediate step $k=' +str(k)+'$')
        plt.plot(flow_results['s_grid'][-1],flow_results['all_norms_computed_in_search'][-1],'-', 
                 label=r'Last step $k=' +str(len(flow_results['s_grid'])-1)+'$')

        x_labels_rounded = [ 0,.5,1 ]
        y_labels_rounded = [ round(y, 1 ) for y in flow_results['minimal_norm_sigma_H_s'][1:3] ]
        y_labels_rounded.insert(0,0)
        plt.xticks(x_labels_rounded)
        plt.xlabel(r'Flow step $s$')
        plt.title(r'Norm change $\vert\vert\sigma(H_{k}))\vert\vert-\vert\vert\sigma(H)\vert\vert$')
        plt.legend(loc='best', bbox_to_anchor=(0.47,0.31)) 
        axin1 = ax.inset_axes([0.08,.45, 0.42, 0.45])
        norms_W_plot = [flow_results['norms_flow_generator_W'][k] for k in range(1,len(flow_results['s_grid']))]
        axin1.plot(np.linspace(1,15,15), norms_W_plot,'-')
        axin1.set_xlabel(r'Flow step $k$')
        axin1.set_title(r'Generator norm $\vert\vert W_k\vert\vert$')
        axin1.set_xticks([5,10,15])
        axin1.grid()
        if save_path is None:
            save_path = 'figs/flow_forwards_N_steps_'+str(self.nmb_flow_steps)+'_L_'+str(self.L)+'.pdf'
        plt.savefig( save_path, format='pdf')
        plt.show()
    
    def get_index_and_step( self, index_stream ): 
        nmb_step = int( np.floor( index_stream ) / 2**self.L ) 
        index_eigenstate = int( index_stream - nmb_step * 2**self.L )
        return index_eigenstate, nmb_step

    def reshape_observables( self, observable ):
        m = int( len(observable) / self.nmb_flow_steps )
        return [ observable[ n * m : (n+1) * m  ]
                for n in range(self.nmb_flow_steps)]
    def observable_over_steps( self, ind, observable ):
        m = int( len(observable) / self.nmb_flow_steps )
        return [ observable[ n * m + ind  ] for n in range(self.nmb_flow_steps)]
    def binary_label_from_state_number( self, state_number):
        return '{0:048b}'.format( state_number )[48-self.L:]
    def find_the_best_eigenstate( self, H ):
        e,U = np.linalg.eigh(H_n) 
        ind = np.argmax(abs(U))
        ind_x = int( ind/ 2**self.L)
        ind_y  = ind - ind_x * 2**self.L
        return [ ind_x, ind_y ]
    
    
    @staticmethod
    def select_indices_of_entries_below_threshold( input_list, threshold ):
        return [ i for i , v in enumerate(input_list) if v < threshold ]
    def select_decoupled_states_from_H( self, H_flowed = None, threshold = 1 ):
        if H_flowed is None:
            if 'final_flowed_H' in self.flow_outputs:
                H_flowed = self.flow_outputs['final_flowed_H'][0]
            else:
                H_flowed = self.H

        norms_couplings_list = []
        for k in range(H_flowed.shape[0]):

            vec = H_flowed[ k, : ] 

            #vec[k] = 0 
            #Modifies H_flowed even in the jupyter notebook! (cf. deepcopy) 
            if self.please_be_exhaustively_verbose is True:
                vec = H_flowed[ k, : ] 
                print(vec,vec[:k],vec[k+1:] , np.concatenate((vec[:k],vec[k+1:])) )

            norms_couplings_list.append( np.linalg.norm( np.concatenate((vec[:k],vec[k+1:])) ) )
            
        selected_eigenstates = self.select_indices_of_entries_below_threshold( norms_couplings_list, threshold )

        if self.please_be_verbose is True:
            print( str(selected_eigenstates) + " are the eigenstates selected with threshold = "  + str(threshold) )

        if self.please_be_exhaustively_verbose is True:
            print( norms_couplings_list )

        if self.please_be_visual is True:
            if self.please_be_exhaustively_visual is True:
                self.show_flowed_H( H = H_flowed )
                plt.show()
            plt.plot( norms_couplings_list )
            plt.plot(selected_eigenstates, [ norms_couplings_list[s] for s in selected_eigenstates], 'o')

        return selected_eigenstates
    def show_selected_couplings( self, selected_couplings = None, H_flowed = None, threshold = 1 ):
        if H_flowed is None:
            if 'final_flowed_H' in self.flow_outputs:
                H_flowed = self.flow_outputs['final_flowed_H'][0]
            else:
                H_flowed = self.H
        if selected_couplings is None:
            selected_couplings = self.select_decoupled_states_from_H( H_flowed = H_flowed, threshold = threshold )
            if self.please_be_verbose is True:
                print( str(selected_couplings) + " are the "+ str(len(selected_couplings)) + " eigenstates selected with threshold = "  + str(threshold) )
        
        N = len( selected_couplings)
        fig = plt.figure()
        for i, ind in enumerate(selected_couplings):  
            plt.subplot( 1, N, i+1 )
            couplings = H_flowed[:, ind]
            plt.plot(couplings)
            plt.plot(ind, couplings[ind],'o')
        plt.show()

    def show_eigenstates( self, selection = None, save_path = None, 
    observable = None, observable_errorbars = None, observable_name = None, observable_errorbars_name = None ):
        self.show_observable_with_errorbars (self, observable = observable, observable_errorbars = observable_errorbars,
        observable_name = 'expected_energy', observable_errorbars_name = 'expected_energy_fluctuation')

    def show_observable_with_errorbars (self, observable = None, observable_errorbars = None,
        observable_name = None, observable_errorbars_name = None ):
        import plotly.graph_objs as go
        import plotly.express as px
        color_discrete_sequence=px.colors.qualitative.G10
        
        if observable_name is not None:
            observable = self.flow_outputs[ observable_name ]
        if observable_fluctuation_name is not None:
            observable_fluctuation = self.flow_outputs[ observable_fluctuation_name ]

        x = [x+1 for x in range( self.nmb_flow_steps)]
        
        Scatter_list = []
        if selection is None:
            selection = range(2**self.L)

        for i, k in enumerate(selection):
            
            e = self.observable_over_steps( k,observable )
            ef = self.observable_over_steps( k, observable_fluctuation )
        
            y = e
            y_upper = [ E+EF for E,EF in zip( e, ef)]
            y_lower = [ E-EF for E,EF in zip( e, ef)]
            state_name = r'$|'+str(self.binary_label_from_state_number(k)) +r'\rangle$'
            Scatter_list.append(
            go.Scatter(
                x=x,
                y=y,
                line=dict(color=color_discrete_sequence[i]),
                mode='lines',
                name= state_name
               
            ) )
            Scatter_list.append(
            go.Scatter(
                x=x+x[::-1], # x, then x reversed
                y=y_upper+y_lower[::-1], # upper, then lower reversed
                fill='toself',
                fillcolor=color_discrete_sequence[i],
                line=dict(color='rgba(0,0,0,0)'),
                    opacity=0.2,
                hoverinfo="skip",
                showlegend=False
            )   
            )
        
        fig = go.Figure(
        Scatter_list
        )
        
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
        fig.show()
        if save_path is not None:
            fig.write_image( save_path )
    def plot_flow_results( flow_results ):
        f=plt.figure(figsize = ( 20,4 ))

        f.add_subplot(1,4,1)

        plt.plot(self.flow_results['s_tot'],self.flow_results['norm_H_mins'], '-o')
        #plt.xticks(s_tot,s_tot)

        chosen_steps = [0,1]
        chosen_steps = range(flow_results['config']['number_flow_steps'])
        f.add_subplot(142)
        for i in chosen_steps:
            plt.plot(flow_results['s_grids'][i],flow_results['norms_H'][i],'-')

        f.add_subplot(143)
        for i in chosen_steps:
            plt.plot(flow_results['s_grids'][i],flow_results['norms_W'][i],'-')

        f.add_subplot(1,4,4)
        plt.plot(flow_results['flow_step_mins'],'.')
        plt.plot(flow_results['s_tot'],'x')
        plt.show()
               
    ### Plot flow evaluation: $\sigma$ decrease, flow step sizes

    def run_param_rc( self, fontsize = 30):
        import matplotlib.pyplot as plt
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['axes.titlesize'] = fontsize
        plt.rcParams['font.size'] = fontsize
        #set_matplotlib_formats('pdf', 'png')
        plt.rcParams['savefig.dpi'] = 75
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['lines.markersize'] = 8
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['legend.labelspacing'] = .3
        plt.rcParams['legend.columnspacing']= .3
        plt.rcParams['legend.handletextpad']= .1
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = "serif"
        plt.rcParams['font.serif'] = "cm"

    ### Plot flow evaluation: $\sigma$ decrease, flow step sizes
    def plot_search_results( flow_results ):
        run_param_rc(15)
        f=plt.figure(figsize = ( 6,4 ))

        f.add_subplot(1,1,1)
        norms = flow_results['initial_norm'] + flow_results['norm_H_for_optimal_Z']
        x_axis = [0]+flow_results['s_tot_for_optimal_Z']
        plt.plot( x_axis, norms, '-o')    
        x_labels_rounded = [ round(x, 2 ) for x in x_axis ]
        x_labels_rounded[0] = 0
        plt.xticks(x_labels_rounded)
        y_labels_rounded = [ round(y, 2 ) for y in norms ]
        plt.yticks(y_labels_rounded)
        plt.grid()
        plt.xlabel(r'Flow step $s$')
        plt.ylabel(r'Norm of off-diagonal $\sigma(H)$')
        save_path = 'figs/flow_Z_optimal_'+str(flow_results['number_flow_steps'])+\
        '_generator_'+str(flow_results['flow_generator_type'])+'.pdf'
        plt.savefig( save_path, format='pdf')
        plt.show()

    def plot_iteration_results( self, flow_results = None ):
        self.run_param_rc(15)
        if flow_results is None:
            flow_results = self.flow_outputs

        f=plt.figure(figsize = ( 20,4 ))
        n=3
        f.add_subplot(1,n,1)
        norms = flow_results['minimal_norm_sigma_H_s']
        x_axis = [sum(flow_results['minimizing_flow_step'][:k])for k in range( 1, len(flow_results['minimizing_flow_step'])+1)]
        
        plt.plot( x_axis, norms, '-o')   
        k=0
        for xy in zip([x+.041 for x in x_axis[0:5]], 
                      [n + 0.05 for n in norms[0:5]]):                                       # <--
            if k == 0:
                plt.annotate('Initial norm', xy=(.05,norms[0]-.2), 
                             textcoords='data')
            elif k == 1:
                plt.annotate('Flow step k = %d' % k, xy=xy, textcoords='data')
        
            else:
                plt.annotate('k = %d' % k, xy=xy, textcoords='data')
            k=k+1
        x_labels_rounded = [ round(x, 2 ) for x in x_axis ]
        x_labels_rounded = [0] + x_labels_rounded[0:5] + [max(x_labels_rounded)]
        x_labels_rounded.pop(3)
        plt.xticks(x_labels_rounded)
        y_labels_rounded = [ round(y, 1 ) for y in norms ]
        y_labels_rounded = y_labels_rounded[0:5] + [min(y_labels_rounded)]
        plt.yticks(y_labels_rounded)
        plt.grid()
        plt.xlabel(r'Flow step $s$')
        plt.title(r'Norm off-diagonal $\vert\vert\sigma(H_k)\vert\vert$')
        a =-.1
        b = 1.05
        plt.annotate('a)', xy = (a,b), xycoords='axes fraction')
        
        chosen_steps = [1,2]
        #chosen_steps = range(flow_results['config']['number_flow_steps'])
        f.add_subplot(1,n,2)
        plt.annotate('b)', xy = (a,b),
                     xycoords='axes fraction')
        ini_norm = flow_results['minimal_norm_sigma_H_s'][0]
        plt.plot(flow_results['s_grid'][1],flow_results['all_norms_computed_in_search'][0]-ini_norm,'-', label=r'First step $k=1$')
        plt.plot(flow_results['s_grid'][2],flow_results['all_norms_computed_in_search'][1]-ini_norm,'-', label=r'Second step $k=2$')
        x_labels_rounded = [0]+[ round(x, 2 ) for x in flow_results['minimizing_flow_step'][1:3] ]
        y_labels_rounded = [ round(y-ini_norm, 1 ) for y in flow_results['minimal_norm_sigma_H_s'][0:2] ]
        y_labels_rounded.insert(0,0)
        plt.xticks(x_labels_rounded)
        plt.yticks(y_labels_rounded)
        plt.grid()
        plt.xlabel(r'Flow step $s$')
        plt.title(r'Norm change $\vert\vert\sigma(H_{k}))\vert\vert-\vert\vert\sigma(H)\vert\vert$')
        plt.legend(loc='best', bbox_to_anchor=(0.8,0.25))
        
        ax = f.add_subplot(1,n,3)
        plt.annotate('c)', xy = (a,b),
                     xycoords='axes fraction')
        ini_norm = flow_results['minimal_norm_sigma_H_s'][0]
        k=11
        plt.plot(flow_results['s_grid'][-1],flow_results['all_norms_computed_in_search'][k],'-', 
                 label=r'Intermediate step $k=' +str(k)+'$')
        plt.plot(flow_results['s_grid'][-1],flow_results['all_norms_computed_in_search'][-1],'-', 
                 label=r'Last step $k=' +str(len(flow_results['s_grid'])-1)+'$')

        x_labels_rounded = [ 0,.5,1 ]
        y_labels_rounded = [ round(y, 1 ) for y in flow_results['minimal_norm_sigma_H_s'][1:3] ]
        y_labels_rounded.insert(0,0)
        plt.xticks(x_labels_rounded)
        #plt.yticks(y_labels_rounded)
        #plt.grid()
        plt.xlabel(r'Flow step $s$')
        plt.title(r'Norm off-diagonal $\vert\vert\sigma(H_{k})\vert\vert$')
        plt.legend(loc='best', bbox_to_anchor=(0.47,0.31)) 
        axin1 = ax.inset_axes([0.08,.45, 0.42, 0.45])
        norms_W_plot = [flow_results['norms_flow_generator_W'][k] for k in range(1,len(flow_results['s_grid']))]
        axin1.plot(np.linspace(1,15,15), norms_W_plot,'-')
        axin1.set_xlabel(r'Flow step $k$')
        axin1.set_title(r'Generator norm $\vert\vert W_k\vert\vert$')
        axin1.set_xticks([5,10,15])
        axin1.grid()

        save_path = 'figs/iterated_flow_N_steps_'+str(flow_results['nmb_flow_steps'])+\
        '_L_'+str(self.L)+'.pdf'
        plt.savefig( save_path, format='pdf')
        plt.show()

    def initialize_flow_results(self):
        self.flow_outputs = {}