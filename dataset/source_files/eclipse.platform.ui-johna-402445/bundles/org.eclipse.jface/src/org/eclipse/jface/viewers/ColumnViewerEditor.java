/*******************************************************************************
 * Copyright (c) 2006, 2013 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *     Tom Schindl <tom.schindl@bestsolution.at> - refactoring (bug 153993)
 *     											   fix in bug: 151295,178946,166500,195908,201906,207676,180504,216706,218336
 *     Tony Juckel -  Bug 130854 - JFace TableViewer ignoring ICellEditor validator state
 *******************************************************************************/

package org.eclipse.jface.viewers;

import org.eclipse.core.runtime.ListenerList;
import org.eclipse.swt.SWT;
import org.eclipse.swt.events.DisposeEvent;
import org.eclipse.swt.events.DisposeListener;
import org.eclipse.swt.events.FocusAdapter;
import org.eclipse.swt.events.FocusEvent;
import org.eclipse.swt.events.FocusListener;
import org.eclipse.swt.events.MouseAdapter;
import org.eclipse.swt.events.MouseEvent;
import org.eclipse.swt.events.MouseListener;
import org.eclipse.swt.events.TraverseEvent;
import org.eclipse.swt.events.TraverseListener;
import org.eclipse.swt.widgets.Control;
import org.eclipse.swt.widgets.Item;

/**
 * This is the base for all editor implementations of Viewers. ColumnViewer
 * implementors have to subclass this class and implement the missing methods
 * @param <E> Type of an single element of the model
 * @param <I> Type of the input
 *
 * @since 3.3
 * @see TableViewerEditor
 * @see TreeViewerEditor
 */
public abstract class ColumnViewerEditor<E,I> {
	private CellEditor cellEditor;

	private ICellEditorListener cellEditorListener;

	private FocusListener focusListener;

	private MouseListener mouseListener;

	private ColumnViewer<E,I> viewer;

	private TraverseListener tabeditingListener;

	private ViewerCell<E> cell;

	private ListenerList editorActivationListener;

	private ColumnViewerEditorActivationStrategy<E,I> editorActivationStrategy;

	private boolean inEditorDeactivation;

	private DisposeListener disposeListener;

	/**
	 * Tabbing from cell to cell is turned off
	 */
	public static final int DEFAULT = 1;

	/**
	 * Should if the end of the row is reach started from the start/end of the
	 * row below/above
	 */
	public static final int TABBING_MOVE_TO_ROW_NEIGHBOR = 1 << 1;

	/**
	 * Should if the end of the row is reach started from the beginning in the
	 * same row
	 */
	public static final int TABBING_CYCLE_IN_ROW = 1 << 2;

	/**
	 * Support tabbing to Cell above/below the current cell
	 */
	public static final int TABBING_VERTICAL = 1 << 3;

	/**
	 * Should tabbing from column to column with in one row be supported
	 */
	public static final int TABBING_HORIZONTAL = 1 << 4;

	/**
	 * Style mask used to enable keyboard activation
	 */
	public static final int KEYBOARD_ACTIVATION = 1 << 5;

	/**
	 * Style mask used to turn <b>off</b> the feature that an editor activation
	 * is canceled on double click. It is also possible to turn off this feature
	 * per cell-editor using {@link CellEditor#getDoubleClickTimeout()}
	 * @since 3.4
	 */
	public static final int KEEP_EDITOR_ON_DOUBLE_CLICK = 1 << 6;

	private int feature;

	/**
	 * @param viewer
	 *            the viewer this editor is attached to
	 * @param editorActivationStrategy
	 *            the strategy used to decide about editor activation
	 * @param feature
	 *            bit mask controlling the editor
	 *            <ul>
	 *            <li>{@link ColumnViewerEditor#DEFAULT}</li>
	 *            <li>{@link ColumnViewerEditor#TABBING_CYCLE_IN_ROW}</li>
	 *            <li>{@link ColumnViewerEditor#TABBING_HORIZONTAL}</li>
	 *            <li>{@link ColumnViewerEditor#TABBING_MOVE_TO_ROW_NEIGHBOR}</li>
	 *            <li>{@link ColumnViewerEditor#TABBING_VERTICAL}</li>
	 *            </ul>
	 */
	protected ColumnViewerEditor(final ColumnViewer<E,I> viewer,
			ColumnViewerEditorActivationStrategy<E,I> editorActivationStrategy,
			int feature) {
		this.viewer = viewer;
		this.editorActivationStrategy = editorActivationStrategy;
		if ((feature & KEYBOARD_ACTIVATION) == KEYBOARD_ACTIVATION) {
			this.editorActivationStrategy
					.setEnableEditorActivationWithKeyboard(true);
		}
		this.feature = feature;
		this.disposeListener = new DisposeListener() {

			public void widgetDisposed(DisposeEvent e) {
				if( viewer.isCellEditorActive() ) {
					cancelEditing();
				}
			}

		};
		initCellEditorListener();
	}

	private void initCellEditorListener() {
		cellEditorListener = new ICellEditorListener() {
			public void editorValueChanged(boolean oldValidState,
					boolean newValidState) {
				// Ignore.
			}

			public void cancelEditor() {
				ColumnViewerEditor.this.cancelEditing();
			}

			public void applyEditorValue() {
				ColumnViewerEditor.this.applyEditorValue();
			}
		};
	}

	private boolean activateCellEditor(final ColumnViewerEditorActivationEvent activationEvent) {

		ViewerColumn<E,I> part = viewer.getViewerColumn(cell.getColumnIndex());
		E element = cell.getElement();

		if (part != null && part.getEditingSupport() != null
				&& part.getEditingSupport().canEdit(element)) {
			cellEditor = part.getEditingSupport().getCellEditor(element);
			if (cellEditor != null) {
				int timeout = cellEditor.getDoubleClickTimeout();

				final int activationTime;

				if (timeout != 0) {
					activationTime = activationEvent.time + timeout;
				} else {
					activationTime = 0;
				}

				if (editorActivationListener != null
						&& !editorActivationListener.isEmpty()) {
					Object[] ls = editorActivationListener.getListeners();
					for (int i = 0; i < ls.length; i++) {
						((ColumnViewerEditorActivationListener) ls[i])
								.beforeEditorActivated(activationEvent);

						// Was the activation canceled ?
						if (activationEvent.cancel) {
							return false;
						}
					}
				}

				updateFocusCell(cell, activationEvent);

				cellEditor.addListener(cellEditorListener);
				part.getEditingSupport().initializeCellEditorValue(cellEditor,
						cell);

				// Tricky flow of control here:
				// activate() can trigger callback to cellEditorListener which
				// will clear cellEditor
				// so must get control first, but must still call activate()
				// even if there is no control.
				final Control control = cellEditor.getControl();
				cellEditor.activate(activationEvent);
				if (control == null) {
					return false;
				}
				setLayoutData(cellEditor.getLayoutData());
				setEditor(control, (Item) cell.getItem(), cell.getColumnIndex());
				cellEditor.setFocus();

				if (cellEditor.dependsOnExternalFocusListener()) {
					if (focusListener == null) {
						focusListener = new FocusAdapter() {
							@Override
							public void focusLost(FocusEvent e) {
								applyEditorValue();
							}
						};
					}
					control.addFocusListener(focusListener);
				}

				mouseListener = new MouseAdapter() {
					@Override
					public void mouseDown(MouseEvent e) {
						// time wrap?
						// check for expiration of doubleClickTime
						if (shouldFireDoubleClick(activationTime, e.time, activationEvent) && e.button == 1) {
							control.removeMouseListener(mouseListener);
							cancelEditing();
							handleDoubleClickEvent();
						} else if (mouseListener != null) {
							control.removeMouseListener(mouseListener);
						}
					}
				};

				if (activationTime != 0
						&& (feature & KEEP_EDITOR_ON_DOUBLE_CLICK) == 0) {
					control.addMouseListener(mouseListener);
				}

				if (tabeditingListener == null) {
					tabeditingListener = new TraverseListener() {

						public void keyTraversed(TraverseEvent e) {
							if ((feature & DEFAULT) != DEFAULT) {
								processTraverseEvent(cell.getColumnIndex(),
										viewer.getViewerRowFromItem(cell
												.getItem()), e);
							}
						}
					};
				}

				control.addTraverseListener(tabeditingListener);

				if (editorActivationListener != null
						&& !editorActivationListener.isEmpty()) {
					Object[] ls = editorActivationListener.getListeners();
					for (int i = 0; i < ls.length; i++) {
						((ColumnViewerEditorActivationListener) ls[i])
								.afterEditorActivated(activationEvent);
					}
				}

				this.cell.getItem().addDisposeListener(disposeListener);

				return true;
			}

		}

		return false;
	}

	private boolean shouldFireDoubleClick(int activationTime, int mouseTime,
			ColumnViewerEditorActivationEvent activationEvent) {
		return mouseTime <= activationTime
				&& activationEvent.eventType != ColumnViewerEditorActivationEvent.KEY_PRESSED
				&& activationEvent.eventType != ColumnViewerEditorActivationEvent.PROGRAMMATIC
				&& activationEvent.eventType != ColumnViewerEditorActivationEvent.TRAVERSAL;
	}

	/**
	 * Applies the current value and deactivates the currently active cell
	 * editor.
	 */
	void applyEditorValue() {
		// avoid re-entering
		if (!inEditorDeactivation) {
			try {
				inEditorDeactivation = true;
				CellEditor c = this.cellEditor;
				if (c != null && this.cell != null) {
					ColumnViewerEditorDeactivationEvent tmp = new ColumnViewerEditorDeactivationEvent(
							cell);
					tmp.eventType = ColumnViewerEditorDeactivationEvent.EDITOR_SAVED;
					if (editorActivationListener != null
							&& !editorActivationListener.isEmpty()) {
						Object[] ls = editorActivationListener.getListeners();
						for (int i = 0; i < ls.length; i++) {

							((ColumnViewerEditorActivationListener) ls[i])
									.beforeEditorDeactivated(tmp);
						}
					}

					Item t = (Item) this.cell.getItem();

					// don't null out table item -- same item is still selected
					if (t != null && !t.isDisposed() && c.isValueValid()) {
						saveEditorValue(c);
					}
					if (!viewer.getControl().isDisposed()) {
						setEditor(null, null, 0);
					}

					c.removeListener(cellEditorListener);
					Control control = c.getControl();
					if (control != null && !control.isDisposed()) {
						if (mouseListener != null) {
							control.removeMouseListener(mouseListener);
							// Clear the instance not needed any more
							mouseListener = null;
						}
						if (focusListener != null) {
							control.removeFocusListener(focusListener);
						}

						if (tabeditingListener != null) {
							control.removeTraverseListener(tabeditingListener);
						}
					}
					c.deactivate(tmp);

					if (editorActivationListener != null
							&& !editorActivationListener.isEmpty()) {
						Object[] ls = editorActivationListener.getListeners();
						for (int i = 0; i < ls.length; i++) {
							((ColumnViewerEditorActivationListener) ls[i])
									.afterEditorDeactivated(tmp);
						}
					}

					if( ! this.cell.getItem().isDisposed() ) {
						this.cell.getItem().removeDisposeListener(disposeListener);
					}
				}

				this.cellEditor = null;
				this.cell = null;
			} finally {
				inEditorDeactivation = false;
			}
		}
	}

	/**
	 * Cancel editing
	 */
	void cancelEditing() {
		// avoid re-entering
		if (!inEditorDeactivation) {
			try {
				inEditorDeactivation = true;
				if (cellEditor != null) {
					ColumnViewerEditorDeactivationEvent tmp = new ColumnViewerEditorDeactivationEvent(
							cell);
					tmp.eventType = ColumnViewerEditorDeactivationEvent.EDITOR_CANCELED;
					if (editorActivationListener != null
							&& !editorActivationListener.isEmpty()) {
						Object[] ls = editorActivationListener.getListeners();
						for (int i = 0; i < ls.length; i++) {

							((ColumnViewerEditorActivationListener) ls[i])
									.beforeEditorDeactivated(tmp);
						}
					}

					if (!viewer.getControl().isDisposed()) {
						setEditor(null, null, 0);
					}

					cellEditor.removeListener(cellEditorListener);

					Control control = cellEditor.getControl();
					if (control != null && !viewer.getControl().isDisposed()) {
						if (mouseListener != null) {
							control.removeMouseListener(mouseListener);
							// Clear the instance not needed any more
							mouseListener = null;
						}
						if (focusListener != null) {
							control.removeFocusListener(focusListener);
						}

						if (tabeditingListener != null) {
							control.removeTraverseListener(tabeditingListener);
						}
					}

					CellEditor oldEditor = cellEditor;
					oldEditor.deactivate(tmp);

					if (editorActivationListener != null
							&& !editorActivationListener.isEmpty()) {
						Object[] ls = editorActivationListener.getListeners();
						for (int i = 0; i < ls.length; i++) {
							((ColumnViewerEditorActivationListener) ls[i])
									.afterEditorDeactivated(tmp);
						}
					}

					if( ! this.cell.getItem().isDisposed() ) {
						this.cell.getItem().addDisposeListener(disposeListener);
					}

					this.cellEditor = null;
					this.cell = null;

				}
			} finally {
				inEditorDeactivation = false;
			}
		}
	}

	/**
	 * Enable the editor by mouse down
	 *
	 * @param event
	 */
	void handleEditorActivationEvent(ColumnViewerEditorActivationEvent event) {

		// Only activate if the event isn't tagged as canceled
		if (!event.cancel
				&& editorActivationStrategy.isEditorActivationEvent(event)) {
			if (cellEditor != null) {
				applyEditorValue();
			}

			@SuppressWarnings("unchecked")
			ViewerCell<E> viewerCell = (ViewerCell<E>) event.getSource();
			this.cell = viewerCell;

			// Only null if we are not in a deactivation process see bug 260892
			if( ! activateCellEditor(event) && ! inEditorDeactivation ) {
				this.cell = null;
				this.cellEditor = null;
			}
		}
	}

	private void saveEditorValue(CellEditor cellEditor) {
		ViewerColumn<E,I> part = viewer.getViewerColumn(cell.getColumnIndex());

		if (part != null && part.getEditingSupport() != null) {
			part.getEditingSupport().saveCellEditorValue(cellEditor, cell);
		}
	}

	/**
	 * Return whether there is an active cell editor.
	 *
	 * @return <code>true</code> if there is an active cell editor; otherwise
	 *         <code>false</code> is returned.
	 */
	boolean isCellEditorActive() {
		return cellEditor != null;
	}

	void handleDoubleClickEvent() {
		viewer.fireDoubleClick(new DoubleClickEvent(viewer, viewer
				.getSelection()));
		viewer.fireOpen(new OpenEvent(viewer, viewer.getSelection()));
	}

	/**
	 * Adds the given listener, it is to be notified when the cell editor is
	 * activated or deactivated.
	 *
	 * @param listener
	 *            the listener to add
	 */
	public void addEditorActivationListener(
			ColumnViewerEditorActivationListener listener) {
		if (editorActivationListener == null) {
			editorActivationListener = new ListenerList();
		}
		editorActivationListener.add(listener);
	}

	/**
	 * Removes the given listener.
	 *
	 * @param listener
	 *            the listener to remove
	 */
	public void removeEditorActivationListener(
			ColumnViewerEditorActivationListener listener) {
		if (editorActivationListener != null) {
			editorActivationListener.remove(listener);
		}
	}

	/**
	 * Process the traverse event and opens the next available editor depending
	 * of the implemented strategy. The default implementation uses the style
	 * constants
	 * <ul>
	 * <li>{@link ColumnViewerEditor#TABBING_MOVE_TO_ROW_NEIGHBOR}</li>
	 * <li>{@link ColumnViewerEditor#TABBING_CYCLE_IN_ROW}</li>
	 * <li>{@link ColumnViewerEditor#TABBING_VERTICAL}</li>
	 * <li>{@link ColumnViewerEditor#TABBING_HORIZONTAL}</li>
	 * </ul>
	 *
	 * <p>
	 * Subclasses may overwrite to implement their custom logic to edit the next
	 * cell
	 * </p>
	 *
	 * @param columnIndex
	 *            the index of the current column
	 * @param row
	 *            the current row - may only be used for the duration of this
	 *            method call
	 * @param event
	 *            the traverse event
	 */
	protected void processTraverseEvent(int columnIndex, ViewerRow<E> row,
			TraverseEvent event) {

		ViewerCell<E> cell2edit = null;

		if (event.detail == SWT.TRAVERSE_TAB_PREVIOUS) {
			event.doit = false;

			if ((event.stateMask & SWT.CTRL) == SWT.CTRL
					&& (feature & TABBING_VERTICAL) == TABBING_VERTICAL) {
				cell2edit = searchCellAboveBelow(row, viewer, columnIndex, true);
			} else if ((feature & TABBING_HORIZONTAL) == TABBING_HORIZONTAL) {
				cell2edit = searchPreviousCell(row, row.getCell(columnIndex),
						row.getCell(columnIndex), viewer);
			}
		} else if (event.detail == SWT.TRAVERSE_TAB_NEXT) {
			event.doit = false;

			if ((event.stateMask & SWT.CTRL) == SWT.CTRL
					&& (feature & TABBING_VERTICAL) == TABBING_VERTICAL) {
				cell2edit = searchCellAboveBelow(row, viewer, columnIndex,
						false);
			} else if ((feature & TABBING_HORIZONTAL) == TABBING_HORIZONTAL) {
				cell2edit = searchNextCell(row, row.getCell(columnIndex), row
						.getCell(columnIndex), viewer);
			}
		}

		if (cell2edit != null) {

			viewer.getControl().setRedraw(false);
			ColumnViewerEditorActivationEvent acEvent = new ColumnViewerEditorActivationEvent(
					cell2edit, event);
			viewer.triggerEditorActivationEvent(acEvent);
			viewer.getControl().setRedraw(true);
		}
	}

	private ViewerCell<E> searchCellAboveBelow(ViewerRow<E> row, ColumnViewer<E,I> viewer,
			int columnIndex, boolean above) {
		ViewerCell<E> rv = null;

		ViewerRow<E> newRow = null;

		if (above) {
			newRow = row.getNeighbor(ViewerRow.ABOVE, false);
		} else {
			newRow = row.getNeighbor(ViewerRow.BELOW, false);
		}

		if (newRow != null) {
			ViewerColumn<E,I> column = viewer.getViewerColumn(columnIndex);
			if (column != null && column.getEditingSupport() != null) {
				@SuppressWarnings("unchecked")
				E element = (E) newRow.getItem().getData();
				if (column.getEditingSupport().canEdit(element)) {
					rv = newRow.getCell(columnIndex);
				} else {
					rv = searchCellAboveBelow(newRow, viewer, columnIndex,
							above);
				}
			} else {
				rv = searchCellAboveBelow(newRow, viewer, columnIndex, above);
			}
		}

		return rv;
	}

	private boolean isCellEditable(ColumnViewer<E,I> viewer, ViewerCell<E> cell) {
		ViewerColumn<E,I> column = viewer.getViewerColumn(cell.getColumnIndex());
		return column != null && column.getEditingSupport() != null
				&& column.getEditingSupport().canEdit(cell.getElement());
	}

	private ViewerCell<E> searchPreviousCell(ViewerRow<E> row,
			ViewerCell<E> currentCell, ViewerCell<E> originalCell, ColumnViewer<E,I> viewer) {
		ViewerCell<E> rv = null;
		ViewerCell<E> previousCell;

		if (currentCell != null) {
			previousCell = currentCell.getNeighbor(ViewerCell.LEFT, true);
		} else {
			if (row.getColumnCount() != 0) {
				previousCell = row.getCell(row.getCreationIndex(row
						.getColumnCount() - 1));
			} else {
				previousCell = row.getCell(0);
			}

		}

		// No endless loop
		if (originalCell.equals(previousCell)) {
			return null;
		}

		if (previousCell != null) {
			if (isCellEditable(viewer, previousCell)) {
				rv = previousCell;
			} else {
				rv = searchPreviousCell(row, previousCell, originalCell, viewer);
			}
		} else {
			if ((feature & TABBING_CYCLE_IN_ROW) == TABBING_CYCLE_IN_ROW) {
				rv = searchPreviousCell(row, null, originalCell, viewer);
			} else if ((feature & TABBING_MOVE_TO_ROW_NEIGHBOR) == TABBING_MOVE_TO_ROW_NEIGHBOR) {
				ViewerRow<E> rowAbove = row.getNeighbor(ViewerRow.ABOVE, false);
				if (rowAbove != null) {
					rv = searchPreviousCell(rowAbove, null, originalCell,
							viewer);
				}
			}
		}

		return rv;
	}

	private ViewerCell<E> searchNextCell(ViewerRow<E> row, ViewerCell<E> currentCell,
			ViewerCell<E> originalCell, ColumnViewer<E,I> viewer) {
		ViewerCell<E> rv = null;

		ViewerCell<E> nextCell;

		if (currentCell != null) {
			nextCell = currentCell.getNeighbor(ViewerCell.RIGHT, true);
		} else {
			nextCell = row.getCell(row.getCreationIndex(0));
		}

		// No endless loop
		if (originalCell.equals(nextCell)) {
			return null;
		}

		if (nextCell != null) {
			if (isCellEditable(viewer, nextCell)) {
				rv = nextCell;
			} else {
				rv = searchNextCell(row, nextCell, originalCell, viewer);
			}
		} else {
			if ((feature & TABBING_CYCLE_IN_ROW) == TABBING_CYCLE_IN_ROW) {
				rv = searchNextCell(row, null, originalCell, viewer);
			} else if ((feature & TABBING_MOVE_TO_ROW_NEIGHBOR) == TABBING_MOVE_TO_ROW_NEIGHBOR) {
				ViewerRow<E> rowBelow = row.getNeighbor(ViewerRow.BELOW, false);
				if (rowBelow != null) {
					rv = searchNextCell(rowBelow, null, originalCell, viewer);
				}
			}
		}

		return rv;
	}

	/**
	 * Position the editor inside the control
	 *
	 * @param w
	 *            the editor control
	 * @param item
	 *            the item (row) in which the editor is drawn in
	 * @param fColumnNumber
	 *            the column number in which the editor is shown
	 */
	protected abstract void setEditor(Control w, Item item, int fColumnNumber);

	/**
	 * set the layout data for the editor
	 *
	 * @param layoutData
	 *            the layout data used when editor is displayed
	 */
	protected abstract void setLayoutData(CellEditor.LayoutData layoutData);

	/**
	 * @param focusCell
	 *            updates the cell with the current input focus
	 * @param event
	 *            the event requesting to update the focusCell
	 */
	protected abstract void updateFocusCell(ViewerCell<E> focusCell,
			ColumnViewerEditorActivationEvent event);

	/**
	 * @return the cell currently holding the focus if no cell has the focus or
	 *         the viewer implementation doesn't support <code>null</code> is
	 *         returned
	 *
	 */
	public ViewerCell<E> getFocusCell() {
		return null;
	}

	/**
	 * @return the viewer working for
	 */
	protected ColumnViewer<E,I> getViewer() {
		return viewer;
	}
}