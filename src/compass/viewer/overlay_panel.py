# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Qt dock listing the annotation groups, with a visibility check box and a
colour swatch each. Tree: store -> layer -> group. Ticking a store or layer
ticks all of its groups. The panel only writes
``ViewerState.visible_groups``; the visuals follow.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from .overlay import Overlay
from .state import ViewerState

_UID_ROLE = QtCore.Qt.ItemDataRole.UserRole


def _swatch(rgba, size: int = 14) -> QtGui.QIcon:
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtGui.QColor.fromRgbF(*rgba))
    return QtGui.QIcon(pm)


class AnnotationPanel(QtWidgets.QDockWidget):
    """Dock widget bound to a :class:`ViewerState` and an :class:`Overlay`."""

    def __init__(self, overlay: Overlay, state: ViewerState, parent=None):
        super().__init__("Annotations", parent)
        self.setObjectName("annotation-panel")
        self._state = state
        self._busy = False
        self._leaves: dict[int, QtWidgets.QTreeWidgetItem] = {}

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # keys stay on the canvas
        self.setWidget(self._tree)

        flag_auto = QtCore.Qt.ItemFlag.ItemIsAutoTristate
        by_source: dict[str, QtWidgets.QTreeWidgetItem] = {}
        by_layer: dict[tuple[str, str], QtWidgets.QTreeWidgetItem] = {}
        for g in overlay.groups:
            src = by_source.get(g.source)
            if src is None:
                src = QtWidgets.QTreeWidgetItem(self._tree, [g.source])
                src.setFlags(src.flags() | flag_auto | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                src.setCheckState(0, QtCore.Qt.CheckState.Checked)
                by_source[g.source] = src
            lay = by_layer.get((g.source, g.layer))
            if lay is None:
                lay = QtWidgets.QTreeWidgetItem(src, [g.layer])
                lay.setFlags(lay.flags() | flag_auto | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                lay.setCheckState(0, QtCore.Qt.CheckState.Checked)
                by_layer[(g.source, g.layer)] = lay
            leaf = QtWidgets.QTreeWidgetItem(lay, [g.label])
            leaf.setFlags((leaf.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                          & ~flag_auto)
            leaf.setCheckState(0, QtCore.Qt.CheckState.Checked)
            leaf.setIcon(0, _swatch(g.color))
            leaf.setData(0, _UID_ROLE, g.uid)
            self._leaves[g.uid] = leaf
        self._tree.expandAll()
        self._tree.itemChanged.connect(self._on_item_changed)
        state.events.visible_groups.connect(self._sync_from_state)

    def _on_item_changed(self, *_) -> None:
        if self._busy:
            return
        shown = frozenset(uid for uid, it in self._leaves.items()
                          if it.checkState(0) == QtCore.Qt.CheckState.Checked)
        self._busy = True
        try:
            self._state.visible_groups = shown
        finally:
            self._busy = False

    def _sync_from_state(self, *_) -> None:
        """Reflect changes made elsewhere (e.g. a script) in the check boxes."""
        if self._busy:
            return
        self._busy = True
        try:
            shown = self._state.visible_groups
            for uid, it in self._leaves.items():
                it.setCheckState(0, QtCore.Qt.CheckState.Checked if uid in shown
                                 else QtCore.Qt.CheckState.Unchecked)
        finally:
            self._busy = False
