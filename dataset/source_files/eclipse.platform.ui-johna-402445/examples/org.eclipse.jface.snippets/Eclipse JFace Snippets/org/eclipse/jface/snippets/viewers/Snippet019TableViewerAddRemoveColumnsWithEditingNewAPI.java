/*******************************************************************************
 * Copyright (c) 2006, 2013 Tom Schindl and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Tom Schindl - initial API and implementation
 *     Hendrik Still <hendrik.still@gammas.de> - bug 417676
 *******************************************************************************/

package org.eclipse.jface.snippets.viewers;


import java.util.ArrayList;
import java.util.List;

import org.eclipse.jface.action.Action;
import org.eclipse.jface.action.IMenuListener;
import org.eclipse.jface.action.IMenuManager;
import org.eclipse.jface.action.MenuManager;
import org.eclipse.jface.util.ConfigureColumns;
import org.eclipse.jface.viewers.CellEditor;
import org.eclipse.jface.viewers.ColumnLabelProvider;
import org.eclipse.jface.viewers.EditingSupport;
import org.eclipse.jface.viewers.IStructuredContentProvider;
import org.eclipse.jface.viewers.TableViewer;
import org.eclipse.jface.viewers.TableViewerColumn;
import org.eclipse.jface.viewers.TextCellEditor;
import org.eclipse.jface.viewers.Viewer;
import org.eclipse.jface.window.SameShellProvider;
import org.eclipse.swt.SWT;
import org.eclipse.swt.events.MouseAdapter;
import org.eclipse.swt.events.MouseEvent;
import org.eclipse.swt.layout.FillLayout;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;

/**
 * Explore the new API added in 3.3 and see how easily you can create reusable
 * components
 *
 * @author Tom Schindl <tom.schindl@bestsolution.at>
 * @since 3.2
 */
public class Snippet019TableViewerAddRemoveColumnsWithEditingNewAPI {

	public class Person {
		public String givenname;

		public String surname;

		public String email;

		public Person(String givenname, String surname, String email) {
			this.givenname = givenname;
			this.surname = surname;
			this.email = email;
		}
	}

	private class MyContentProvider implements IStructuredContentProvider<Person,List<Person>> {

		public Person[] getElements(List<Person> inputElement) {
			Person[] persons = new Person[inputElement.size()];
			return inputElement.toArray(persons);
		}

		public void dispose() {
		}

		public void inputChanged(Viewer<? extends List<Person>> viewer, List<Person> oldInput, List<Person> newInput) {

		}

	}



	private class GivenNameLabelProvider extends ColumnLabelProvider<Person,List<Person>> {
		public String getText(Person element) {
			return element.givenname;
		}
	}

	private class GivenNameEditing extends EditingSupport {
		private TextCellEditor cellEditor;

		public GivenNameEditing(TableViewer<Person,List<Person>> viewer) {
			super(viewer);
			cellEditor = new TextCellEditor(viewer.getTable());
		}

		protected boolean canEdit(Object element) {
			return true;
		}

		protected CellEditor getCellEditor(Object element) {
			return cellEditor;
		}

		protected Object getValue(Object element) {
			return ((Person) element).givenname;
		}

		protected void setValue(Object element, Object value) {
			((Person) element).givenname = value.toString();
			getViewer().update(element, null);
		}
	}

	private class SurNameLabelProvider extends ColumnLabelProvider<Person,List<Person>>{
		public String getText(Person element) {
			return element.surname;
		}
	}

	private class SurNameEditing extends EditingSupport {
		private TextCellEditor cellEditor;

		public SurNameEditing( TableViewer viewer ) {
			super(viewer);
			cellEditor = new TextCellEditor(viewer.getTable());
		}

		protected boolean canEdit(Object element) {
			return true;
		}

		protected CellEditor getCellEditor(Object element) {
			return cellEditor;
		}

		protected Object getValue(Object element) {
			return ((Person) element).surname;
		}

		protected void setValue(Object element, Object value) {
			((Person) element).surname = value.toString();
			getViewer().update(element, null);
		}
	}

	private class EmailLabelProvider extends ColumnLabelProvider<Person,List<Person>> {
		public String getText(Person element) {
			return element.email;
		}
	}

	private class EmailEditing extends EditingSupport {
		private TextCellEditor cellEditor;

		public EmailEditing( TableViewer<Person,List<Person>> viewer ) {
			super(viewer);
			cellEditor = new TextCellEditor(viewer.getTable());
		}

		protected boolean canEdit(Object element) {
			return true;
		}

		protected CellEditor getCellEditor(Object element) {
			return cellEditor;
		}

		protected Object getValue(Object element) {
			return ((Person) element).email;
		}

		protected void setValue(Object element, Object value) {
			((Person) element).email = value.toString();
			getViewer().update(element, null);
		}
	}

	private int activeColumn = -1;

	private TableViewerColumn<Person,List<Person>> column;

	public Snippet019TableViewerAddRemoveColumnsWithEditingNewAPI(Shell shell) {
		final TableViewer<Person,List<Person>> v = new TableViewer<Person,List<Person>>(shell, SWT.BORDER
				| SWT.FULL_SELECTION);

		TableViewerColumn<Person,List<Person>> column = new TableViewerColumn<Person,List<Person>>(v,SWT.NONE);
		column.setLabelProvider(new GivenNameLabelProvider());
		column.setEditingSupport(new GivenNameEditing(v));

		column.getColumn().setWidth(200);
		column.getColumn().setText("Givenname");
		column.getColumn().setMoveable(true);

		column = new TableViewerColumn<Person,List<Person>>(v,SWT.NONE);
		column.setLabelProvider(new SurNameLabelProvider());
		column.setEditingSupport(new SurNameEditing(v));
		column.getColumn().setWidth(200);
		column.getColumn().setText("Surname");
		column.getColumn().setMoveable(true);

		List<Person> model = createModel();

		v.setContentProvider(new MyContentProvider());
		v.setInput(model);
		v.getTable().setLinesVisible(true);
		v.getTable().setHeaderVisible(true);

		addMenu(v);
		triggerColumnSelectedColumn(v);
	}

	private void triggerColumnSelectedColumn(final TableViewer<Person,List<Person>> v) {
		v.getTable().addMouseListener(new MouseAdapter() {

			public void mouseDown(MouseEvent e) {
				int x = 0;
				for (int i = 0; i < v.getTable().getColumnCount(); i++) {
					x += v.getTable().getColumn(i).getWidth();
					if (e.x <= x) {
						activeColumn = i;
						break;
					}
				}
			}

		});
	}

	private void removeEmailColumn(TableViewer<Person,List<Person>> v) {
		column.getColumn().dispose();
		v.refresh();
	}

	private void addEmailColumn(TableViewer<Person,List<Person>> v, int columnIndex) {
		column = new TableViewerColumn<Person,List<Person>>(v, SWT.NONE, columnIndex);
		column.setLabelProvider(new EmailLabelProvider());
		column.setEditingSupport(new EmailEditing(v));
		column.getColumn().setText("E-Mail");
		column.getColumn().setResizable(false);

		v.refresh();

		column.getColumn().setWidth(200);

	}

	private void addMenu(final TableViewer<Person,List<Person>> v) {
		final MenuManager mgr = new MenuManager();

		final Action insertEmailBefore = new Action("Insert E-Mail before") {
			public void run() {
				addEmailColumn(v, activeColumn);
			}
		};

		final Action insertEmailAfter = new Action("Insert E-Mail after") {
			public void run() {
				addEmailColumn(v, activeColumn + 1);
			}
		};

		final Action removeEmail = new Action("Remove E-Mail") {
			public void run() {
				removeEmailColumn(v);
			}
		};

		final Action configureColumns = new Action("Configure Columns...") {
			public void run() {
				ConfigureColumns.forTable(v.getTable(), new SameShellProvider(v.getControl()));
			}
		};

		mgr.setRemoveAllWhenShown(true);
		mgr.addMenuListener(new IMenuListener() {

			public void menuAboutToShow(IMenuManager manager) {
				if (v.getTable().getColumnCount() == 2) {
					manager.add(insertEmailBefore);
					manager.add(insertEmailAfter);
				} else {
					manager.add(removeEmail);
				}
				manager.add(configureColumns);
			}

		});

		v.getControl().setMenu(mgr.createContextMenu(v.getControl()));
	}

	private List<Person> createModel() {

		List<Person> persons = new ArrayList<Person>();
		persons.add(new Person("Tom", "Schindl", "tom.schindl@bestsolution.at"));
		persons.add(new Person("Boris", "Bokowski",
				"boris_bokowski@ca.ibm.com"));
		persons.add(new Person("Tod", "Creasey", "tod_creasey@ca.ibm.com"));

		return persons;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Display display = new Display();

		Shell shell = new Shell(display);
		shell.setLayout(new FillLayout());
		new Snippet019TableViewerAddRemoveColumnsWithEditingNewAPI(shell);
		shell.open();

		while (!shell.isDisposed()) {
			if (!display.readAndDispatch())
				display.sleep();
		}

		display.dispose();

	}

}
