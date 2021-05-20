/*******************************************************************************
 * Copyright (c) 2007 - 2013 Tom Schindl and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Tom Schindl - initial API and implementation
 *     Lars Vogel (lars.vogel@gmail.com) - Bug 413427
 *     Hendrik Still <hendrik.still@gammas.de> - bug 417676
 *******************************************************************************/

package org.eclipse.jface.snippets.viewers;

import java.util.ArrayList;
import java.util.List;

import org.eclipse.jface.resource.ImageDescriptor;
import org.eclipse.jface.resource.JFaceResources;
import org.eclipse.jface.viewers.CellEditor;
import org.eclipse.jface.viewers.CheckboxCellEditor;
import org.eclipse.jface.viewers.ComboBoxCellEditor;
import org.eclipse.jface.viewers.ICellEditorListener;
import org.eclipse.jface.viewers.ICellModifier;
import org.eclipse.jface.viewers.IStructuredContentProvider;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.jface.viewers.LabelProvider;
import org.eclipse.jface.viewers.StructuredViewer;
import org.eclipse.jface.viewers.TableViewer;
import org.eclipse.jface.viewers.TextCellEditor;
import org.eclipse.jface.viewers.Viewer;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Image;
import org.eclipse.swt.layout.FillLayout;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Control;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.swt.widgets.Table;
import org.eclipse.swt.widgets.TableColumn;
import org.eclipse.swt.widgets.TableItem;

/**
 * Snippet to present editor different CellEditors within one column in 3.2
 * for 3.3 and above please use the new EditingSupport class
 *
 * @author Tom Schindl <tom.schindl@bestsolution.at>
 *
 *
 */
public class Snippet033CellEditorPerRowPre33 {
	private class MyCellModifier implements ICellModifier {

		private TableViewer viewer;

		public void setViewer(TableViewer viewer) {
			this.viewer = viewer;
		}

		/*
		 * (non-Javadoc)
		 *
		 * @see org.eclipse.jface.viewers.ICellModifier#canModify(java.lang.Object,
		 *      java.lang.String)
		 */
		public boolean canModify(Object element, String property) {
			return true;
		}

		/*
		 * (non-Javadoc)
		 *
		 * @see org.eclipse.jface.viewers.ICellModifier#getValue(java.lang.Object,
		 *      java.lang.String)
		 */
		public Object getValue(Object element, String property) {
			if( element instanceof MyModel3 ) {
				return new Boolean(((MyModel3)element).checked);
			} else if( element instanceof MyModel2 ) {
				return new Integer(((MyModel) element).counter);
			} else {
				return ((MyModel) element).counter + "";
			}
		}

		/*
		 * (non-Javadoc)
		 *
		 * @see org.eclipse.jface.viewers.ICellModifier#modify(java.lang.Object,
		 *      java.lang.String, java.lang.Object)
		 */
		public void modify(Object element, String property, Object value) {
			TableItem item = (TableItem) element;

			if( item.getData() instanceof MyModel3 ) {
				((MyModel3) item.getData()).checked=((Boolean)value).booleanValue();
			} else {
				((MyModel) item.getData()).counter = Integer.parseInt(value
						.toString());
			}

			viewer.update(item.getData(), null);
		}
	}

	private class MyContentProvider implements IStructuredContentProvider<MyModel,List<MyModel>> {

		/*
		 * (non-Javadoc)
		 *
		 * @see org.eclipse.jface.viewers.IStructuredContentProvider#getElements(java.lang.Object)
		 */
		public MyModel[] getElements(List<MyModel> inputElement) {
			MyModel[] myModels = new MyModel[inputElement.size()];
			return inputElement.toArray(myModels);
		}

		/*
		 * (non-Javadoc)
		 *
		 * @see org.eclipse.jface.viewers.IContentProvider#dispose()
		 */
		public void dispose() {

		}

		/*
		 * (non-Javadoc)
		 *
		 * @see org.eclipse.jface.viewers.IContentProvider#inputChanged(org.eclipse.jface.viewers.Viewer,
		 *      java.lang.Object, java.lang.Object)
		 */
		public void inputChanged(Viewer<? extends List<MyModel>> viewer, List<MyModel> oldInput, List<MyModel> newInput) {

		}

	}

	public class MyModel {
		public int counter;

		public MyModel(int counter) {
			this.counter = counter;
		}

		public String toString() {
			return "Item " + this.counter;
		}
	}

	public class MyModel2 extends MyModel {

		public MyModel2(int counter) {
			super(counter);
		}

		public String toString() {
			return "Special Item " + this.counter;
		}
	}

	public class MyModel3 extends MyModel {
		public boolean checked;

		public MyModel3(int counter) {
			super(counter);
		}

		public String toString() {
			return "Special Item " + this.counter;
		}
	}


	public class DelegatingEditor extends CellEditor {

		private final StructuredViewer viewer;

		private final CellEditor delegatingTextEditor;

		private final CellEditor delegatingDropDownEditor;

		private CellEditor activeEditor;

		private final CellEditor delegatingCheckBoxEditor;

		private class DelegatingListener implements ICellEditorListener {

			public void applyEditorValue() {
				fireApplyEditorValue();
			}

			public void cancelEditor() {
				fireCancelEditor();
			}

			public void editorValueChanged(boolean oldValidState,
					boolean newValidState) {
				fireEditorValueChanged(oldValidState, newValidState);
			}

		}

		public DelegatingEditor(StructuredViewer viewer, Composite parent) {
			super(parent);
			this.viewer = viewer;
			DelegatingListener l = new DelegatingListener();
			this.delegatingTextEditor = new TextCellEditor(parent);
			this.delegatingTextEditor.addListener(l);

			String[] elements = new String[10];

			for (int i = 0; i < 10; i++) {
				elements[i] = i+"";
			}

			this.delegatingDropDownEditor = new ComboBoxCellEditor(parent,elements);
			this.delegatingDropDownEditor.addListener(l);

			this.delegatingCheckBoxEditor = new CheckboxCellEditor(parent);
			this.delegatingCheckBoxEditor.addListener(l);
		}

		protected Control createControl(Composite parent) {
			return null;
		}

		protected Object doGetValue() {
			return activeEditor.getValue();
		}

		protected void doSetFocus() {
			activeEditor.setFocus();
		}

		public void activate() {
			if( activeEditor != null ) {
				activeEditor.activate();
			}
		}

		protected void doSetValue(Object value) {

			if( ((IStructuredSelection)this.viewer.getSelection()).getFirstElement() instanceof MyModel3 ) {
				activeEditor = delegatingCheckBoxEditor;
			} else if( ((IStructuredSelection)this.viewer.getSelection()).getFirstElement() instanceof MyModel2 ) {
				activeEditor = delegatingDropDownEditor;
			} else {
				activeEditor = delegatingTextEditor;
			}

			activeEditor.setValue(value);
		}

		public void deactivate() {
			if( activeEditor != null ) {
				Control control = activeEditor.getControl();
				if (control != null && !control.isDisposed()) {
					control.setVisible(false);
				}
			}
		}

		public void dispose() {

		}

		public Control getControl() {
			return activeEditor.getControl();
		}
	}

	public Snippet033CellEditorPerRowPre33(Shell shell) {
		final Table table = new Table(shell, SWT.BORDER | SWT.FULL_SELECTION);
		final MyCellModifier modifier = new MyCellModifier();

		final TableViewer<MyModel,List<MyModel>> v = new TableViewer<MyModel,List<MyModel>>(table);
		modifier.setViewer(v);

		TableColumn column = new TableColumn(table, SWT.NONE);
		column.setWidth(200);

		v.setLabelProvider(new MyLabelProvider());
		v.setContentProvider(new MyContentProvider());
		v.setCellModifier(modifier);
		v.setColumnProperties(new String[] { "column1" });
		v.setCellEditors(new CellEditor[] { new DelegatingEditor(v,v.getTable()) });

		List<MyModel> model = createModel();
		v.setInput(model);
		v.getTable().setLinesVisible(true);
	}

	private class MyLabelProvider extends LabelProvider<MyModel> {
		public Image getImage(MyModel element) {
			if( element instanceof MyModel3 ) {
				if( ((MyModel3)element).checked ) {
					return JFaceResources.getImage("IMG_1");
				} else {
					return JFaceResources.getImage("IMG_2");
				}

			}
			return super.getImage(element);
		}

	}

	private List<MyModel> createModel() {
		List<MyModel> elements = new ArrayList<MyModel>(30);

		for (int i = 0; i < 10; i++) {
			elements.add(i,new MyModel3(i));
		}

		for (int i = 0; i < 10; i++) {
			elements.add(i+10,new MyModel(i));
		}

		for (int i = 0; i < 10; i++) {
			elements.add(i+20,new MyModel2(i));
		}

		return elements;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Display display = new Display();
		JFaceResources.getImageRegistry().put("IMG_1", ImageDescriptor.createFromURL(Snippet033CellEditorPerRowPre33.class.getClassLoader().getResource("org/eclipse/jface/snippets/dialogs/cancel.png")));
		JFaceResources.getImageRegistry().put("IMG_2", ImageDescriptor.createFromURL(Snippet033CellEditorPerRowPre33.class.getClassLoader().getResource("org/eclipse/jface/snippets/dialogs/filesave.png")));
		Shell shell = new Shell(display);
		shell.setLayout(new FillLayout());
		new Snippet033CellEditorPerRowPre33(shell);
		shell.open();

		while (!shell.isDisposed()) {
			if (!display.readAndDispatch())
				display.sleep();
		}

		display.dispose();

	}

}
