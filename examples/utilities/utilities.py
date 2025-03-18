# -*- coding: utf-8 -*-
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygfunction.borefield import Borefield
from pygfunction.gfunction import gFunction
from pygfunction.pipes import SingleUTube, MultipleUTube, Coaxial


def initialize_figure() -> Figure:
    """
    Initialize a matplotlib figure object with overwritten default parameters.

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    plt.rc('font', size=9)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('lines', lw=1.5, markersize=5.0)
    plt.rc('savefig', dpi=500)
    fig = plt.figure()
    return fig


def format_axes(ax: Axes):
    """
    Adjust axis parameters.

    Parameters
    ----------
    ax : axis
        Axis object (matplotlib).

    """
    from matplotlib.ticker import AutoMinorLocator
    # Draw major and minor tick marks inwards
    ax.tick_params(
        axis='both', which='both', direction='in',
        bottom=True, top=True, left=True, right=True)
    # Auto-adjust minor tick marks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return


def format_axes_3d(ax: Axes):
    """
    Adjust axis parameters.

    Parameters
    ----------
    ax : axis
        Axis object (matplotlib).

    """
    # Draw major and minor tick marks inwards
    ax.tick_params(
        axis='both', which='major', direction='in',
        bottom=True, top=True, left=True, right=True)
    # Auto-adjust minor tick marks
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.zaxis.set_minor_locator(AutoMinorLocator())
    return


def visualize_g_function(g_function: gFunction, which=None) -> Figure:
    """
    Plot the g-function of the borefield.

    Parameters
    ----------
    g_function : gFunction
    which : list of tuple, optional
        Tuples (i, j) of the variable mass flow rate g-functions to plot.
        If None, all g-functions are plotted.
        Default is None.

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    # Configure figure and axes
    fig = initialize_figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'ln$(t/t_s)$')
    ax.set_ylabel(r'$g$-function')
    format_axes(ax)

    # Borefield characteristic time
    ts = np.mean([b.H for b in g_function.boreholes]) ** 2 / (9. * g_function.alpha)
    # Dimensionless time (log)
    lntts = np.log(g_function.time / ts)
    # Draw g-function
    if g_function.solver.nMassFlow == 0:
        ax.plot(lntts, g_function.gFunc)
    elif which is None:
        for j in range(g_function.solver.nMassFlow):
            for i in range(g_function.solver.nMassFlow):
                ax.plot(
                    lntts,
                    g_function.gFunc[i, j, :],
                    label=f'$g_{{{i}{j}}}$')
        plt.legend()
    else:
        if which is None:
            which = [
                (i, j) for j in range(g_function.solver.nMassFlow)
                for i in range(g_function.solver.nMassFlow)]
        for (i, j) in which:
            ax.plot(
                lntts,
                g_function.gFunc[i, j, :],
                label=f'$g_{{{i}{j}}}$')
        plt.legend()

    # Adjust figure to window
    plt.tight_layout()
    return fig


def visualize_field(
        borefield: Borefield, viewTop: bool = True, view3D: bool = True,
        labels: bool = True, showTilt: bool = True) -> Figure:
    """
    Plot the top view and 3D view of borehole positions.

    Parameters
    ----------
    borefield : Borefield
    viewTop : bool, optional
        Set to True to plot top view.
        Default is True
    view3D : bool, optional
        Set to True to plot 3D view.
        Default is True
    labels : bool, optional
        Set to True to annotate borehole indices to top view plot.
        Default is True
    showTilt : bool, optional
        Set to True to show borehole inclination on top view plot.
        Default is True

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    # Configure figure and axes
    fig = initialize_figure()
    if viewTop and view3D:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')
    elif viewTop:
        ax1 = fig.add_subplot(111)
    elif view3D:
        ax2 = fig.add_subplot(111, projection='3d')
    if viewTop:
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        format_axes(ax1)
    if view3D:
        ax2.set_xlabel(r'$x$ [m]')
        ax2.set_ylabel(r'$y$ [m]')
        ax2.set_zlabel(r'$z$ [m]')
        format_axes_3d(ax2)
        ax2.invert_zaxis()

    # Bottom end of boreholes
    x_H = borefield.x + borefield.H * np.sin(borefield.tilt) * np.cos(borefield.orientation)
    y_H = borefield.y + borefield.H * np.sin(borefield.tilt) * np.sin(borefield.orientation)
    z_H = borefield.D + borefield.H * np.cos(borefield.tilt)

    # -------------------------------------------------------------------------
    # Top view
    # -------------------------------------------------------------------------
    if viewTop:
        if showTilt:
            ax1.plot(
                np.stack((borefield.x, x_H), axis=0),
                np.stack((borefield.y, y_H), axis=0),
                'k--')
        ax1.plot(borefield.x, borefield.y, 'ko')
        if labels:
            for i, borehole in enumerate(borefield):
                ax1.text(
                    borehole.x,
                    borehole.y,
                    f' {i}',
                    ha="left",
                    va="bottom")

    # -------------------------------------------------------------------------
    # 3D view
    # -------------------------------------------------------------------------
    if view3D:
        ax2.plot(borefield.x, borefield.y, borefield.D, 'ko')
        for i in range(borefield.nBoreholes):
            ax2.plot(
                (borefield.x[i], x_H[i]),
                (borefield.y[i], y_H[i]),
                (borefield.D[i], z_H[i]),
                'k-')

    if viewTop and view3D:
        plt.tight_layout(rect=[0, 0.0, 0.90, 1.0])
    else:
        plt.tight_layout()

    return fig


def visualize_pipes(pipe: Union[SingleUTube, MultipleUTube]) -> Figure:
    """
    Plot the cross-section view of the borehole.

    Parameters
    ----------
    pipe : SingleUTube | MultipleUTube

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    # Configure figure and axes
    fig = initialize_figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.axis('equal')
    format_axes(ax)

    # Color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    lw = plt.rcParams['lines.linewidth']

    # Borehole wall outline
    ax.plot([-pipe.b.r_b, 0., pipe.b.r_b, 0.],
            [0., pipe.b.r_b, 0., -pipe.b.r_b],
            'k.', alpha=0.)
    borewall = plt.Circle(
        (0., 0.), radius=pipe.b.r_b, fill=False,
        color='k', linestyle='--', lw=lw)
    ax.add_patch(borewall)

    # Pipes
    for i in range(pipe.nPipes):
        # Coordinates of pipes
        (x_in, y_in) = pipe.pos[i]
        (x_out, y_out) = pipe.pos[i + pipe.nPipes]

        # Pipe outline (inlet)
        pipe_in_in = plt.Circle(
            (x_in, y_in), radius=pipe.r_in,
            fill=False, linestyle='-', color=colors[i], lw=lw)
        pipe_in_out = plt.Circle(
            (x_in, y_in), radius=pipe.r_out,
            fill=False, linestyle='-', color=colors[i], lw=lw)
        ax.text(x_in, y_in, i, ha="center", va="center")

        # Pipe outline (outlet)
        pipe_out_in = plt.Circle(
            (x_out, y_out), radius=pipe.r_in,
            fill=False, linestyle='-', color=colors[i], lw=lw)
        pipe_out_out = plt.Circle(
            (x_out, y_out), radius=pipe.r_out,
            fill=False, linestyle='-', color=colors[i], lw=lw)
        ax.text(x_out, y_out, i + pipe.nPipes,
                ha="center", va="center")

        ax.add_patch(pipe_in_in)
        ax.add_patch(pipe_in_out)
        ax.add_patch(pipe_out_in)
        ax.add_patch(pipe_out_out)

    plt.tight_layout()

    return fig


def visualize_coaxial_pipes(pipe: Coaxial) -> Figure:
    """
    Plot the cross-section view of the borehole.

    Parameters
    ----------
    pipe : Coaxial

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    # Configure figure and axes
    fig = initialize_figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.axis('equal')
    format_axes(ax)

    # Color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    lw = plt.rcParams['lines.linewidth']

    # Borehole wall outline
    ax.plot([-pipe.b.r_b, 0., pipe.b.r_b, 0.],
            [0., pipe.b.r_b, 0., -pipe.b.r_b],
            'k.', alpha=0.)
    borewall = plt.Circle(
        (0., 0.), radius=pipe.b.r_b, fill=False,
        color='k', linestyle='--', lw=lw)
    ax.add_patch(borewall)

    # Pipes
    for i, (pos, color) in enumerate(zip(pipe.pos, colors)):
        # Coordinates of pipes
        (x_in, y_in) = pos
        (x_out, y_out) = pos

        # Pipe outline (inlet)
        pipe_in_in = plt.Circle(
            (x_in, y_in), radius=pipe.r_in[0],
            fill=False, linestyle='-', color=color, lw=lw)
        pipe_in_out = plt.Circle(
            (x_in, y_in), radius=pipe.r_out[0],
            fill=False, linestyle='-', color=color, lw=lw)
        if pipe._iInner == 0:
            ax.text(x_in, y_in, i, ha="center", va="center")
        else:
            ax.text(x_in + 0.5 * (pipe.r_out[0] + pipe.r_in[1]), y_in, i,
                    ha="center", va="center")

        # Pipe outline (outlet)
        pipe_out_in = plt.Circle(
            (x_out, y_out), radius=pipe.r_in[1],
            fill=False, linestyle='-', color=color, lw=lw)
        pipe_out_out = plt.Circle(
            (x_out, y_out), radius=pipe.r_out[1],
            fill=False, linestyle='-', color=color, lw=lw)
        if pipe._iInner == 1:
            ax.text(x_out, y_out, i + pipe.nPipes, ha="center", va="center")
        else:
            ax.text(x_out + 0.5 * (pipe.r_out[0] + pipe.r_in[1]), y_out,
                    i + pipe.nPipes, ha="center", va="center")

        ax.add_patch(pipe_in_in)
        ax.add_patch(pipe_in_out)
        ax.add_patch(pipe_out_in)
        ax.add_patch(pipe_out_out)

    plt.tight_layout()

    return fig


def visualize_temperatures(
        g_function: gFunction, iBoreholes=None, showTilt=True, which=None) -> Figure:
    """
    Plot the time-variation of the average borehole wall temperatures.

    Parameters
    ----------
    g_function : gFunction
    iBoreholes : list of int
        Borehole indices to plot temperatures.
        If iBoreholes is None, temperatures are plotted for all boreholes.
        Default is None.
    showTilt : bool
        Set to True to show borehole inclination.
        Default is True
    which : list of int, optional
        Indices i of the diagonal variable mass flow rate g-functions for
        which to plot borehole wall temperatures.
        If None, all diagonal g-functions are plotted.
        Default is None.

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    # If iBoreholes is None, then plot all boreholes
    if iBoreholes is None:
        iBoreholes = range(len(g_function.solver.boreholes))
    # Import temperatures
    T_b = g_function._temperatures(iBoreholes)
    # Borefield characteristic time
    ts = np.mean([b.H for b in g_function.solver.boreholes])**2/(9.*g_function.alpha)
    # Dimensionless time (log)
    lntts = np.log(g_function.time/ts)


    if g_function.solver.nMassFlow == 0:
        # Configure figure and axes
        fig = initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'ln$(t/t_s)$')
        ax2.set_ylabel(r'$\bar{T}_b$')
        format_axes(ax2)
        # Plot curves for requested boreholes
        for i, borehole in enumerate(g_function.solver.boreholes):
            if i in iBoreholes:
                # Draw borehole wall temperature
                line = ax2.plot(lntts, T_b[iBoreholes.index(i)])
                color = line[-1]._color
                # Draw colored marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color=color)
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color=color)
            else:
                # Draw black marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color='k')
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color='k')

        # Adjust figure to window
        plt.tight_layout()
    else:
        m_flow = g_function.solver.m_flow
        if which is None:
            which = [n for n in range(g_function.solver.nMassFlow)]
        for n in which:
            # Configure figure and axes
            fig = initialize_figure()
            fig.suptitle(
                f'Borehole wall temperatures for m_flow={m_flow[n]} kg/s')
            ax1 = fig.add_subplot(121)
            ax1.set_xlabel(r'$x$ [m]')
            ax1.set_ylabel(r'$y$ [m]')
            ax1.axis('equal')
            format_axes(ax1)
            ax2 = fig.add_subplot(122)
            ax2.set_xlabel(r'ln$(t/t_s)$')
            ax2.set_ylabel(r'$\bar{T}_b$')
            format_axes(ax2)
            # Plot curves for requested boreholes
            for i, borehole in enumerate(g_function.solver.boreholes):
                if i in iBoreholes:
                    # Draw borehole wall temperature
                    line = ax2.plot(lntts, T_b[iBoreholes.index(i)][n])
                    color = line[-1]._color
                    # Draw colored marker for borehole position
                    if showTilt:
                        ax1.plot(
                            [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                            [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                             linestyle='--',
                             marker='None',
                             color=color)
                    ax1.plot(borehole.x,
                             borehole.y,
                             linestyle='None',
                             marker='o',
                             color=color)
                else:
                    # Draw black marker for borehole position
                    if showTilt:
                        ax1.plot(
                            [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                            [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                             linestyle='--',
                             marker='None',
                             color='k')
                    ax1.plot(borehole.x,
                             borehole.y,
                             linestyle='None',
                             marker='o',
                             color='k')

            # Adjust figure to window
            plt.tight_layout()
    return fig


def visualize_temperature_profiles(
        g_function: gFunction, time=None, iBoreholes=None, showTilt=True, which=None) -> Figure:
    """
    Plot the borehole wall temperature profiles at chosen time.

    Parameters
    ----------
    g_function : gFunction
    time : float
        Values of time (in seconds) to plot temperature profiles.
        If time is None, temperatures are plotted at the last time step.
        Default is None.
    iBoreholes : list of int
        Borehole indices to plot temperature profiles.
        If iBoreholes is None, temperatures are plotted for all boreholes.
        Default is None.
    showTilt : bool
        Set to True to show borehole inclination.
        Default is True
    which : list of int, optional
        Indices i of the diagonal variable mass flow rate g-functions for
        which to plot borehole wall temperatures.
        If None, all diagonal g-functions are plotted.
        Default is None.

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    # If iBoreholes is None, then plot all boreholes
    if iBoreholes is None:
        iBoreholes = range(len(g_function.boreholes))
    # Import temperature profiles
    z, T_b = g_function._temperature_profiles(time, iBoreholes)

    if g_function.solver.nMassFlow == 0:
        # Configure figure and axes
        fig = initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'$T_b$')
        ax2.set_ylabel(r'$z$ [m]')
        ax2.invert_yaxis()
        format_axes(ax2)

        # Plot curves for requested boreholes
        for i, borehole in enumerate(g_function.solver.boreholes):
            if i in iBoreholes:
                # Draw borehole wall temperature profile
                line = ax2.plot(
                    T_b[iBoreholes.index(i)],
                    z[iBoreholes.index(i)])
                color = line[-1]._color
                # Draw colored marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color=color)
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color=color)
            else:
                # Draw black marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color='k')
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color='k')

        plt.tight_layout()
    else:
        m_flow = g_function.solver.m_flow
        if which is None:
            which = [n for n in range(g_function.solver.nMassFlow)]
        for n in which:
            # Configure figure and axes
            fig = initialize_figure()
            fig.suptitle(
            f'Borehole wall temperature profiles for m_flow={m_flow[n]} kg/s')
            ax1 = fig.add_subplot(121)
            ax1.set_xlabel(r'$x$ [m]')
            ax1.set_ylabel(r'$y$ [m]')
            ax1.axis('equal')
            format_axes(ax1)
            ax2 = fig.add_subplot(122)
            ax2.set_xlabel(r'$T_b$')
            ax2.set_ylabel(r'$z$ [m]')
            ax2.invert_yaxis()
            format_axes(ax2)

            # Plot curves for requested boreholes
            for i, borehole in enumerate(g_function.solver.boreholes):
                if i in iBoreholes:
                    # Draw borehole wall temperature profile
                    line = ax2.plot(
                        T_b[iBoreholes.index(i)][n],
                        z[iBoreholes.index(i)])
                    color = line[-1]._color
                    # Draw colored marker for borehole position
                    if showTilt:
                        ax1.plot(
                            [borehole.x, borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                            [borehole.y, borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
                             linestyle='--',
                             marker='None',
                             color=color)
                    ax1.plot(borehole.x,
                             borehole.y,
                             linestyle='None',
                             marker='o',
                             color=color)
                else:
                    # Draw black marker for borehole position
                    if showTilt:
                        ax1.plot(
                            [borehole.x, borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                            [borehole.y, borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
                             linestyle='--',
                             marker='None',
                             color='k')
                    ax1.plot(borehole.x,
                             borehole.y,
                             linestyle='None',
                             marker='o',
                             color='k')

            plt.tight_layout()
    return fig


def visualize_heat_extraction_rates(
        g_function: gFunction, iBoreholes=None, showTilt=True, which=None) -> Figure:
    """
    Plot the time-variation of the average heat extraction rates.

    Parameters
    ----------
    g_function : gFunction
    iBoreholes : list of int
        Borehole indices to plot heat extraction rates.
        If iBoreholes is None, heat extraction rates are plotted for all
        boreholes.
        Default is None.
    showTilt : bool
        Set to True to show borehole inclination.
        Default is True
    which : list of int, optional
        Indices i of the diagonal variable mass flow rate g-functions for
        which to plot heat extraction rates.
        If None, all diagonal g-functions are plotted.
        Default is None.

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    # If iBoreholes is None, then plot all boreholes
    if iBoreholes is None:
        iBoreholes = range(len(g_function.solver.boreholes))
    # Import heat extraction rates
    Q_t = g_function._heat_extraction_rates(iBoreholes)
    # Borefield characteristic time
    ts = np.mean([b.H for b in g_function.solver.boreholes]) ** 2 / (9. * g_function.alpha)
    # Dimensionless time (log)
    lntts = np.log(g_function.time / ts)

    if g_function.solver.nMassFlow == 0:
        # Configure figure and axes
        fig = initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'ln$(t/t_s)$')
        ax2.set_ylabel(r'$\bar{Q}_b$')
        format_axes(ax2)

        # Plot curves for requested boreholes
        for i, borehole in enumerate(g_function.solver.boreholes):
            if i in iBoreholes:
                # Draw heat extraction rate
                line = ax2.plot(lntts, Q_t[iBoreholes.index(i)])
                color = line[-1]._color
                # Draw colored marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x,
                         borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                        [borehole.y,
                         borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
                        linestyle='--',
                        marker='None',
                        color=color)
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color=color)
            else:
                # Draw black marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x,
                         borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                        [borehole.y,
                         borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
                        linestyle='--',
                        marker='None',
                        color='k')
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color='k')

        # Adjust figure to window
        plt.tight_layout()
    else:
        m_flow = g_function.solver.m_flow
        if which is None:
            which = [n for n in range(g_function.solver.nMassFlow)]
        for n in which:
            # Configure figure and axes
            fig = initialize_figure()
            fig.suptitle(
                f'Heat extraction rates for m_flow={m_flow[n]} kg/s')
            ax1 = fig.add_subplot(121)
            ax1.set_xlabel(r'$x$ [m]')
            ax1.set_ylabel(r'$y$ [m]')
            ax1.axis('equal')
            format_axes(ax1)
            ax2 = fig.add_subplot(122)
            ax2.set_xlabel(r'ln$(t/t_s)$')
            ax2.set_ylabel(r'$\bar{Q}_b$')
            format_axes(ax2)

            # Plot curves for requested boreholes
            for i, borehole in enumerate(g_function.solver.boreholes):
                if i in iBoreholes:
                    # Draw heat extraction rate
                    line = ax2.plot(lntts, Q_t[iBoreholes.index(i)][n])
                    color = line[-1]._color
                    # Draw colored marker for borehole position
                    if showTilt:
                        ax1.plot(
                            [borehole.x,
                             borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                            [borehole.y,
                             borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
                            linestyle='--',
                            marker='None',
                            color=color)
                    ax1.plot(borehole.x,
                             borehole.y,
                             linestyle='None',
                             marker='o',
                             color=color)
                else:
                    # Draw black marker for borehole position
                    if showTilt:
                        ax1.plot(
                            [borehole.x,
                             borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                            [borehole.y,
                             borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
                            linestyle='--',
                            marker='None',
                            color='k')
                    ax1.plot(borehole.x,
                             borehole.y,
                             linestyle='None',
                             marker='o',
                             color='k')

            # Adjust figure to window
            plt.tight_layout()

    return fig


def visualize_heat_extraction_rate_profiles(
        g_function: gFunction, time=None, iBoreholes=None, showTilt=True, which=None) -> Figure:
    """
    Plot the heat extraction rate profiles at chosen time.

    Parameters
    ----------
    g_function : gFunction
    time : float
        Values of time (in seconds) to plot heat extraction rate profiles.
        If time is None, heat extraction rates are plotted at the last
        time step.
        Default is None.
    iBoreholes : list of int
        Borehole indices to plot heat extraction rate profiles.
        If iBoreholes is None, heat extraction rates are plotted for all
        boreholes.
        Default is None.
    showTilt : bool
        Set to True to show borehole inclination.
        Default is True
    which : list of int, optional
        Indices i of the diagonal variable mass flow rate g-functions for
        which to plot heat extraction rates.
        If None, all diagonal g-functions are plotted.
        Default is None.

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    # If iBoreholes is None, then plot all boreholes
    if iBoreholes is None:
        iBoreholes = range(len(g_function.solver.boreholes))
    # Import heat extraction rate profiles
    z, Q_b = g_function._heat_extraction_rate_profiles(time, iBoreholes)

    if g_function.solver.nMassFlow == 0:
        # Configure figure and axes
        fig = initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'$Q_b$')
        ax2.set_ylabel(r'$z$ [m]')
        ax2.invert_yaxis()
        format_axes(ax2)

        # Plot curves for requested boreholes
        for i, borehole in enumerate(g_function.solver.boreholes):
            if i in iBoreholes:
                # Draw heat extraction rate profile
                line = ax2.plot(
                    Q_b[iBoreholes.index(i)], z[iBoreholes.index(i)])
                color = line[-1]._color
                # Draw colored marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color=color)
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color=color)
            else:
                # Draw black marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color='k')
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color='k')

        # Adjust figure to window
        plt.tight_layout()
    else:
        m_flow = g_function.solver.m_flow
        if which is None:
            which = [n for n in range(g_function.solver.nMassFlow)]
        for n in which:
            # Configure figure and axes
            fig = initialize_figure()
            fig.suptitle(
                f'Heat extraction rate profiles for m_flow={m_flow[n]} kg/s')
            ax1 = fig.add_subplot(121)
            ax1.set_xlabel(r'$x$ [m]')
            ax1.set_ylabel(r'$y$ [m]')
            ax1.axis('equal')
            format_axes(ax1)
            ax2 = fig.add_subplot(122)
            ax2.set_xlabel(r'$Q_b$')
            ax2.set_ylabel(r'$z$ [m]')
            ax2.invert_yaxis()
            format_axes(ax2)

            # Plot curves for requested boreholes
            for i, borehole in enumerate(g_function.solver.boreholes):
                if i in iBoreholes:
                    # Draw heat extraction rate profile
                    line = ax2.plot(
                        Q_b[iBoreholes.index(i)][n],
                        z[iBoreholes.index(i)])
                    color = line[-1]._color
                    # Draw colored marker for borehole position
                    if showTilt:
                        ax1.plot(
                            [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                            [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                             linestyle='--',
                             marker='None',
                             color=color)
                    ax1.plot(borehole.x,
                             borehole.y,
                             linestyle='None',
                             marker='o',
                             color=color)
                else:
                    # Draw black marker for borehole position
                    if showTilt:
                        ax1.plot(
                            [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                            [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                             linestyle='--',
                             marker='None',
                             color='k')
                    ax1.plot(borehole.x,
                             borehole.y,
                             linestyle='None',
                             marker='o',
                             color='k')

            # Adjust figure to window
            plt.tight_layout()

    return fig
